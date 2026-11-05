#!/usr/bin/env python3
"""Read-only, low-overhead flight recorder for the independently owned DeepSeek service.

The recorder never calls the inference service, imports MLX, or changes the
target process.  It observes the qualified service PID through its PID file,
samples public process/host counters with the already-proven native reader,
and watches the existing private request archive.  A slow completed request
freezes the scalar ring into an immutable minimal evidence bundle.  Request
and response payloads are never copied from the archive.  Automatic enrichment
is disabled in this prototype.

This file is an isolated prototype.  It is not installed or started by merely
running its tests or ``--check-config``.
"""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shlex
import signal
import stat
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from zoneinfo import ZoneInfo


from portable import HERE, REPO, state_dir
ROOT = REPO
DEFAULT_ARCHIVE = state_dir() / "logs/requests"
DEFAULT_PID_FILE = state_dir() / "process.json"
DEFAULT_OUT = state_dir() / "logs/flight-recorder"
DEFAULT_SERVICE_LOG = state_dir() / "logs/service.log"
DEFAULT_API_KEY_FILE = state_dir() / "api-key"
NATIVE_READER = HERE / "telemetry/private_os_telemetry_thermal.py"
LOCAL_TZ = ZoneInfo("Asia/Shanghai")
GIB = 1024**3
PROCESSED_JOURNAL_LIMIT = 10_000
ARCHIVE_SETTLE_SECONDS = 5.0
PUBLICATION_OVERLAP_NS = 5_000_000_000
MAX_METADATA_BYTES = 2 * 1024 * 1024
MAX_ARCHIVE_RESPONSE_BYTES = 8 * 1024 * 1024
MAX_STATE_BYTES = 16 * 1024 * 1024
MAX_PID_FILE_BYTES = 64
MAX_API_KEY_BYTES = 64 * 1024

# Keep the expensive query narrow.  The kernel IOGPU record has processID=0;
# attribution comes from its actor name plus the request time window, never by
# pretending it carries the Python PID.
UNIFIED_PREDICATE = " OR ".join(
    (
        '(process == "kernel" AND senderImagePath CONTAINS[c] "IOGPUFamily" '
        'AND eventMessage CONTAINS[c] "FAILED To Free")',
        'eventMessage CONTAINS[c] "compressor_exhausted"',
        'eventMessage CONTAINS[c] "low-swap"',
        'eventMessage CONTAINS[c] "greater than 50% compressed memory"',
        'process == "thermalmonitord"',
        'eventMessage CONTAINS[c] "thermal pressure"',
        'eventMessage CONTAINS[c] "performance warning"',
    )
)

ACCELERATOR_RE = re.compile(
    r"<(?P<actor>[^>]+)> FAILED To Free.*?total accelerator = "
    r"0x(?P<total>[0-9A-Fa-f]+) max = 0x(?P<maximum>[0-9A-Fa-f]+)"
)
INCIDENT_NAME_RE = re.compile(
    r"^(?:\.partial-)?\d{8}-\d{6}-[0-9a-f]{32}(?:-failed)?$"
)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_owned_regular(path: Path, *, max_bytes: int) -> tuple[bytes, os.stat_result]:
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        info = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                or info.st_nlink != 1):
            raise RuntimeError(f"input must be an owned, singly-linked regular file: {path}")
        if info.st_size > max_bytes:
            raise RuntimeError(f"input exceeds {max_bytes} bytes: {path}")
        chunks = []
        remaining = max_bytes + 1
        while remaining > 0:
            chunk = os.read(fd, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        data = b"".join(chunks)
        if len(data) > max_bytes:
            raise RuntimeError(f"input exceeds {max_bytes} bytes: {path}")
        return data, info
    finally:
        os.close(fd)


def ensure_private_directory(path: Path) -> None:
    """Create or tighten an owned directory without following a leaf symlink."""
    try:
        info = path.lstat()
    except FileNotFoundError:
        path.mkdir(mode=0o700, parents=True, exist_ok=False)
        info = path.lstat()
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise RuntimeError(f"private output path is not a real directory: {path}")
    if info.st_uid != os.getuid():
        raise PermissionError(f"private output directory has another owner: {path}")
    os.chmod(path, 0o700, follow_symlinks=False)


def verify_private_regular_fd(fd: int, *, label: str) -> None:
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_nlink != 1:
        raise RuntimeError(f"{label} must be an owned, singly-linked regular file")
    os.fchmod(fd, 0o600)


def reject_unsafe_existing_file(path: Path) -> None:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return
    if (stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.getuid() or info.st_nlink != 1):
        raise RuntimeError(f"refusing unsafe existing private file: {path}")


def atomic_json(path: Path, value: object) -> None:
    ensure_private_directory(path.parent)
    reject_unsafe_existing_file(path)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    fd = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o600,
    )
    try:
        verify_private_regular_fd(fd, label="atomic JSON temporary")
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        try:
            os.close(fd)
        except OSError:
            pass
        Path(temporary).unlink(missing_ok=True)
        raise
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def append_private(path: Path, value: object) -> None:
    ensure_private_directory(path.parent)
    fd = os.open(
        path,
        os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW,
        0o600,
    )
    try:
        verify_private_regular_fd(fd, label="append target")
    except BaseException:
        os.close(fd)
        raise
    with os.fdopen(fd, "a") as stream:
        stream.write(json.dumps(value, separators=(",", ":"), ensure_ascii=False) + "\n")


def canonical_uuid(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError):
        return None
    canonical = str(parsed)
    return canonical if value == canonical else None


def valid_processed_key(value: object) -> bool:
    return canonical_uuid(value) is not None or (
        isinstance(value, str)
        and re.fullmatch(r"invalid-name-sha256:[0-9a-f]{64}", value) is not None
    )


class NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Reject redirects before urllib can construct a second authorized request."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def load_native_reader():
    spec = importlib.util.spec_from_file_location("omlx_flight_native", NATIVE_READER)
    if spec is None or spec.loader is None:
        raise RuntimeError("native telemetry reader is unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.NativeReader(), module


@dataclass(frozen=True)
class Config:
    archive_root: str
    pid_file: str
    output_root: str
    service_log: str = str(DEFAULT_SERVICE_LOG)
    api_key_file: str = str(DEFAULT_API_KEY_FILE)
    base_url: str = "http://127.0.0.1:8080"
    poll_seconds: float = 2.0
    archive_scan_seconds: float = 15.0
    state_checkpoint_seconds: float = 30.0
    prune_seconds: float = 900.0
    ring_seconds: int = 1800
    postroll_seconds: int = 30
    log_preroll_seconds: int = 30
    slow_tps: float = 30.0
    minimum_completion_tokens: int = 64
    minimum_generation_seconds: float = 5.0
    spool_hours: int = 6
    incident_days: int = 30
    max_incidents: int = 20
    max_log_bytes: int = 16 * 1024 * 1024
    max_service_log_slice_bytes: int = 2 * 1024 * 1024
    max_snapshot_bytes: int = 8 * 1024 * 1024
    max_event_file_bytes: int = 4 * 1024 * 1024
    event_file_rotations: int = 4
    finalize_retries: int = 3
    trigger_cooldown_seconds: int = 900
    max_pending_incidents: int = 1
    enable_incident_enrichment: bool = False
    enable_target_subprocess_snapshots: bool = False
    enable_service_log_slice: bool = False

    def validate(self) -> None:
        archive = Path(self.archive_root)
        pid_file = Path(self.pid_file)
        output = Path(self.output_root)
        service_log = Path(self.service_log)
        api_key_file = Path(self.api_key_file)
        if any(not path.is_absolute() for path in (archive, pid_file, output, service_log, api_key_file)):
            raise ValueError("all paths must be absolute")
        if self.enable_incident_enrichment or self.enable_target_subprocess_snapshots:
            raise ValueError("Portable recorder performs metadata-only observation")
        if not 1.0 <= self.poll_seconds <= 10.0:
            raise ValueError("poll_seconds must be between 1 and 10")
        if not 5.0 <= self.archive_scan_seconds <= 300.0:
            raise ValueError("archive_scan_seconds must be between 5 and 300")
        if not 10.0 <= self.state_checkpoint_seconds <= 300.0:
            raise ValueError("state_checkpoint_seconds must be between 10 and 300")
        if not 60.0 <= self.prune_seconds <= 3600.0:
            raise ValueError("prune_seconds must be between 60 and 3600")
        if not 300 <= self.ring_seconds <= 7200:
            raise ValueError("ring_seconds must be between 300 and 7200")
        if not 0 <= self.postroll_seconds <= 120:
            raise ValueError("postroll_seconds must be between 0 and 120")
        if not 1 <= self.slow_tps <= 1000:
            raise ValueError("slow_tps is out of range")
        if not 16 <= self.minimum_completion_tokens <= 4096:
            raise ValueError("minimum_completion_tokens is out of range")
        if not 1 <= self.minimum_generation_seconds <= 600:
            raise ValueError("minimum_generation_seconds is out of range")
        if not 1 <= self.spool_hours <= 24 or not 1 <= self.incident_days <= 365:
            raise ValueError("retention is out of range")
        if not 1 <= self.max_incidents <= 100 or not 1024 <= self.max_log_bytes <= 64 * 1024**2:
            raise ValueError("incident bounds are out of range")
        if not 64 * 1024 <= self.max_service_log_slice_bytes <= 8 * 1024**2:
            raise ValueError("service log slice bound is out of range")
        if not 64 * 1024 <= self.max_snapshot_bytes <= 32 * 1024**2:
            raise ValueError("snapshot bound is out of range")
        if not 64 * 1024 <= self.max_event_file_bytes <= 16 * 1024**2:
            raise ValueError("event file bound is out of range")
        if not 1 <= self.event_file_rotations <= 10 or not 1 <= self.finalize_retries <= 5:
            raise ValueError("rotation/retry bounds are out of range")
        if not 600 <= self.trigger_cooldown_seconds <= 3600:
            raise ValueError("trigger cooldown must be between 10 and 60 minutes")
        if self.max_pending_incidents != 1:
            raise ValueError("production admission requires a hard pending cap of one")
        if (type(self.enable_incident_enrichment) is not bool
                or type(self.enable_target_subprocess_snapshots) is not bool
                or type(self.enable_service_log_slice) is not bool):
            raise ValueError("snapshot flags must be booleans")
        if self.enable_incident_enrichment:
            raise ValueError(
                "automatic incident enrichment is not production-qualified; use the manual workflow"
            )
        if self.enable_target_subprocess_snapshots or self.enable_service_log_slice:
            raise ValueError(
                "snapshot and service-log options require unavailable incident enrichment"
            )


def request_scalar_record(
    directory: Path, *, metadata_bytes: bytes | None = None,
    response_bytes: bytes | None = None,
) -> dict | None:
    """Return privacy-bounded request metadata and timing, if complete."""
    metadata_path = directory / "metadata.json"
    response_path = directory / "response.json"
    try:
        if metadata_bytes is None:
            metadata_bytes, _ = read_owned_regular(metadata_path, max_bytes=MAX_METADATA_BYTES)
        if response_bytes is None:
            response_bytes, _ = read_owned_regular(
                response_path, max_bytes=MAX_ARCHIVE_RESPONSE_BYTES
            )
        metadata = json.loads(metadata_bytes)
        response = json.loads(response_bytes)
        if not isinstance(metadata, dict) or not isinstance(response, dict):
            return None
        directory_id = canonical_uuid(directory.name)
        metadata_id = canonical_uuid(metadata.get("request_id"))
        if directory_id is None or metadata_id != directory_id:
            return None
        if metadata.get("status") not in {"completed", "http_error"}:
            return None
        usage = response.get("usage")
        if not isinstance(usage, dict):
            completed = response.get("completed_response")
            usage = completed.get("usage") if isinstance(completed, dict) else None
        if not isinstance(usage, dict):
            return None
        generation_s = usage.get("generation_duration")
        tps = usage.get("generation_tokens_per_second")
        completion = usage.get("completion_tokens", usage.get("output_tokens"))
        prompt = usage.get("prompt_tokens", usage.get("input_tokens"))
        numeric_required = (generation_s, tps, completion)
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               or not math.isfinite(float(value)) for value in numeric_required):
            return None
        if (float(generation_s) < 0 or float(tps) < 0
                or not isinstance(completion, int) or isinstance(completion, bool)
                or completion < 0):
            return None
        finished = metadata.get("finished_unix")
        started = metadata.get("started_unix")
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               or not math.isfinite(float(value)) for value in (started, finished)):
            return None
        if (float(finished) < float(started)
                or float(generation_s) > float(finished) - float(started) + 1.0):
            return None
        datetime.fromtimestamp(float(started), LOCAL_TZ)
        datetime.fromtimestamp(float(finished), LOCAL_TZ)
        prompt_details = usage.get("prompt_tokens_details")
        if prompt_details is not None and not isinstance(prompt_details, dict):
            return None
        cached = (prompt_details or {}).get("cached_tokens")
        if (cached is not None and
                (not isinstance(cached, int) or isinstance(cached, bool) or cached < 0)):
            return None
        if (prompt is not None and
                (not isinstance(prompt, int) or isinstance(prompt, bool) or prompt < 0)):
            return None
        engine_ids = metadata.get("engine_request_ids") or []
        if not isinstance(engine_ids, list) or not all(isinstance(item, str) for item in engine_ids):
            return None
        client_id = metadata.get("client_request_id") or ""
        if not isinstance(client_id, str):
            return None
        return {
            "archive_request_id": directory_id,
            "engine_request_ids": list(engine_ids),
            "client_request_id_sha256": hashlib.sha256(client_id.encode()).hexdigest(),
            "path": metadata.get("path") if isinstance(metadata.get("path"), str) else None,
            "status": metadata.get("status"),
            "http_status": metadata.get("http_status"),
            "transport_complete": metadata.get("transport_complete"),
            "started_unix": float(started),
            "finished_unix": float(finished),
            "request_bytes": metadata.get("request_bytes"),
            "response_bytes": metadata.get("response_bytes"),
            "request_sha256": metadata.get("request_sha256"),
            "response_sha256": metadata.get("response_sha256"),
            "prompt_tokens": int(prompt) if prompt is not None else None,
            "cached_tokens": int(cached) if cached is not None else None,
            "completion_tokens": int(completion),
            "generation_duration_s": float(generation_s),
            "generation_tokens_per_second": float(tps),
            "time_to_first_token_s": usage.get("time_to_first_token"),
            "prompt_eval_duration_s": usage.get("prompt_eval_duration"),
            "response_usage_keys": sorted(usage),
            "source_files": {
                "metadata": {
                    "path": str(metadata_path), "sha256": hashlib.sha256(metadata_bytes).hexdigest(),
                },
                "response": {
                    "path": str(response_path), "sha256": hashlib.sha256(response_bytes).hexdigest(),
                },
            },
        }
    except (FileNotFoundError, json.JSONDecodeError, OSError, TypeError, ValueError,
            AttributeError, OverflowError):
        return None


def inspect_archive_entry(directory: Path, *, now: float) -> dict:
    """Classify one archive directory without allowing a bad terminal row to loop."""
    canonical_id = canonical_uuid(directory.name)
    safe_key = canonical_id or (
        "invalid-name-sha256:" + hashlib.sha256(directory.name.encode(errors="replace")).hexdigest()
    )
    metadata_path = directory / "metadata.json"
    try:
        metadata_bytes, metadata_stat = read_owned_regular(
            metadata_path, max_bytes=MAX_METADATA_BYTES
        )
    except FileNotFoundError:
        return {"state": "incomplete", "key": safe_key}
    except OSError as error:
        return {
            "state": "error", "key": safe_key,
            "reason": type(error).__name__,
            "publication_mtime_ns": int(now * 1_000_000_000),
        }
    if metadata_stat.st_mtime > now + ARCHIVE_SETTLE_SECONDS:
        # A future-dated archive entry must not advance the durable cursor and
        # suppress normally published entries until wall time catches up.
        return {
            "state": "error", "key": safe_key,
            "reason": "metadata_mtime_is_in_the_future",
            "publication_mtime_ns": metadata_stat.st_mtime_ns,
        }
    settled = now - metadata_stat.st_mtime >= ARCHIVE_SETTLE_SECONDS
    if canonical_id is None:
        return ({
            "state": "terminal_invalid", "key": safe_key,
            "reason": "archive_directory_name_is_not_canonical_uuid",
            "publication_mtime_ns": metadata_stat.st_mtime_ns,
        } if settled else {"state": "incomplete", "key": safe_key})
    try:
        metadata = json.loads(metadata_bytes)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return ({
            "state": "terminal_invalid", "key": safe_key,
            "reason": "metadata_json_invalid",
            "publication_mtime_ns": metadata_stat.st_mtime_ns,
        } if settled else {"state": "incomplete", "key": safe_key})
    except OSError as error:
        return {
            "state": "error", "key": safe_key,
            "reason": type(error).__name__,
            "publication_mtime_ns": metadata_stat.st_mtime_ns,
        }
    if not isinstance(metadata, dict):
        return ({
            "state": "terminal_invalid", "key": safe_key,
            "reason": "metadata_schema_not_object",
            "publication_mtime_ns": metadata_stat.st_mtime_ns,
        } if settled else {"state": "incomplete", "key": safe_key})
    status = metadata.get("status")
    if status not in {"completed", "http_error"}:
        if isinstance(status, str):
            return {"state": "incomplete", "key": safe_key}
        return ({
            "state": "terminal_invalid", "key": safe_key,
            "reason": "metadata_status_invalid",
            "publication_mtime_ns": metadata_stat.st_mtime_ns,
        } if settled else {"state": "incomplete", "key": safe_key})
    # Invalid terminal rows advance only by the trusted file timestamp.  An
    # untrusted far-future finished_unix must never poison the durable watermark.
    publication_mtime_ns = metadata_stat.st_mtime_ns
    response_path = directory / "response.json"
    try:
        response_bytes, response_stat = read_owned_regular(
            response_path, max_bytes=MAX_ARCHIVE_RESPONSE_BYTES
        )
    except FileNotFoundError:
        return ({
            "state": "terminal_invalid", "key": safe_key,
            "reason": "terminal_response_missing",
            "publication_mtime_ns": publication_mtime_ns,
        } if settled else {"state": "incomplete", "key": safe_key})
    except OSError as error:
        return {
            "state": "error", "key": safe_key,
            "reason": type(error).__name__,
            "publication_mtime_ns": publication_mtime_ns,
        }
    record = request_scalar_record(
        directory, metadata_bytes=metadata_bytes, response_bytes=response_bytes
    )
    if record is not None:
        if record["finished_unix"] > now + ARCHIVE_SETTLE_SECONDS:
            return {
                "state": "terminal_invalid", "key": safe_key,
                "reason": "terminal_finished_time_is_in_the_future",
                "publication_mtime_ns": publication_mtime_ns,
            }
        return {
            "state": "complete", "key": canonical_id,
            "record": record, "publication_mtime_ns": publication_mtime_ns,
        }
    latest_mtime = max(metadata_stat.st_mtime, response_stat.st_mtime)
    if now - latest_mtime < ARCHIVE_SETTLE_SECONDS:
        return {"state": "incomplete", "key": safe_key}
    return {
        "state": "terminal_invalid", "key": safe_key,
        "reason": "terminal_archive_schema_invalid",
        "publication_mtime_ns": publication_mtime_ns,
    }


def trigger_reasons(record: dict, config: Config) -> list[str]:
    reasons = []
    if (
        record["completion_tokens"] >= config.minimum_completion_tokens
        and record["generation_duration_s"] >= config.minimum_generation_seconds
        and record["generation_tokens_per_second"] < config.slow_tps
    ):
        reasons.append("completed_decode_below_absolute_tps_floor")
    return reasons


def compact_sample(row: dict, *, observed_pid: int) -> dict:
    process = row.get("process") or {}
    host = row.get("host_vm") or {}
    swap = row.get("swap_usage") or {}
    sysctl = row.get("sysctl") or {}
    thermal = row.get("system_thermal") or {}
    scalar = {}
    for name in (
        "user_time", "system_time", "pageins", "wired_size", "resident_size",
        "phys_footprint", "proc_start_abstime", "lifetime_max_phys_footprint",
        "diskio_bytesread", "diskio_byteswritten",
    ):
        scalar[name] = process.get(name)
    vm = {name: host.get(name) for name in (
        "page_bytes", "free_count", "active_count", "inactive_count", "wire_count",
        "pageins", "pageouts", "decompressions", "compressions", "swapins", "swapouts",
        "compressor_page_count", "total_uncompressed_pages_in_compressor", "swapped_count",
        "throttled_count",
    )}
    pressure = (sysctl.get("kern.memorystatus_vm_pressure_level") or {}).get("value")
    return {
        "observed_at_unix": time.time(),
        "observed_monotonic_ns": time.monotonic_ns(),
        "pid": observed_pid,
        "process": scalar,
        "host_vm": vm,
        "swap": {
            "total_bytes": swap.get("total_bytes"),
            "available_bytes": swap.get("available_bytes"),
            "used_bytes": swap.get("used_bytes"),
        },
        "memory_pressure_raw": pressure,
        "system_thermal": {"raw": thermal.get("raw"), "state": thermal.get("state")},
    }


def file_position(path: Path) -> dict:
    try:
        info = path.stat()
    except OSError as error:
        return {"available": False, "error_type": type(error).__name__}
    return {
        "available": True,
        "device": info.st_dev,
        "inode": info.st_ino,
        "size": info.st_size,
        "mtime_ns": info.st_mtime_ns,
    }


def runtime_is_idle(profile: dict, cache: dict, expected_pid: int) -> tuple[bool, list[str]]:
    reasons = []
    if profile.get("ready") is not True:
        reasons.append("runtime_not_ready")
    if profile.get("pid") != expected_pid:
        reasons.append("runtime_pid_changed")
    actual = profile.get("actual") or profile.get("settings") or {}
    if actual.get("running_requests") != 0:
        reasons.append("runtime_running_requests_nonzero_or_unavailable")
    schedulers = [row.get("scheduler") or {} for row in (cache.get("models") or {}).values()]
    if not schedulers:
        reasons.append("scheduler_snapshot_unavailable")
    elif any(any(row.get(key) != 0 for key in
                 ("running_requests", "waiting_requests", "prefilling_requests"))
             for row in schedulers):
        reasons.append("scheduler_not_idle")
    return not reasons, reasons


def canonical_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def relocate_paths(value: object, old: Path, new: Path) -> object:
    """Rewrite staging paths before atomically publishing an incident."""
    if isinstance(value, dict):
        return {key: relocate_paths(item, old, new) for key, item in value.items()}
    if isinstance(value, list):
        return [relocate_paths(item, old, new) for item in value]
    if isinstance(value, str) and value.startswith(str(old)):
        return str(new) + value[len(str(old)):]
    return value


def verify_service_identity(pid):
    from control import owns, read_record
    record = read_record()
    if not owns(record) or record["pid"] != pid:
        raise RuntimeError("Service PID identity is not owned by this deployment")
    return {"verified": True, "pid": pid, "created": record["created"],
            "verified_at_unix": time.time()}


def log_time(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S%z")


def read_ndjson(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(errors="replace").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and row.get("eventType") == "logEvent":
            rows.append(row)
    return rows


def summarize_unified_log(rows: list[dict], decode_start: float, decode_end: float) -> dict:
    iogpu = []
    memory = []
    thermal = []
    malformed_rows = 0
    for row in rows:
        if not isinstance(row, dict):
            malformed_rows += 1
            continue
        message = str(row.get("eventMessage") or "")
        match = ACCELERATOR_RE.search(message)
        if match:
            try:
                timestamp = datetime.strptime(
                    row["timestamp"], "%Y-%m-%d %H:%M:%S.%f%z"
                ).timestamp()
            except (KeyError, TypeError, ValueError):
                malformed_rows += 1
                continue
            iogpu.append({
                "timestamp": row.get("timestamp"),
                "epoch": timestamp,
                "actor": match.group("actor"),
                "total_accelerator_bytes": int(match.group("total"), 16),
                "max_accelerator_bytes": int(match.group("maximum"), 16),
                "inside_estimated_decode": decode_start <= timestamp <= decode_end,
            })
        lowered = message.lower()
        if any(term in lowered for term in ("compressor_exhausted", "low-swap", "greater than 50% compressed memory")):
            memory.append({"timestamp": row.get("timestamp"), "event_message": message})
        if row.get("processImagePath", "").endswith("thermalmonitord") or "thermal pressure" in lowered or "performance warning" in lowered:
            thermal.append({"timestamp": row.get("timestamp"), "message_type": row.get("messageType"), "event_message": message})
    python_rows = [row for row in iogpu if row["actor"].startswith("python")]
    decode_python = [row for row in python_rows if row["inside_estimated_decode"]]
    totals = [row["total_accelerator_bytes"] for row in decode_python]
    maxima = [row["max_accelerator_bytes"] for row in decode_python]
    duration = max(decode_end - decode_start, 1e-9)
    return {
        "event_count": len(rows),
        "malformed_row_count": malformed_rows,
        "iogpu_failed_to_free": {
            "all_actors": len(iogpu),
            "python_actor": len(python_rows),
            "python_actor_inside_estimated_decode": len(decode_python),
            "python_actor_rate_per_second_inside_estimated_decode": len(decode_python) / duration,
            "decode_total_accelerator_bytes_range": [min(totals), max(totals)] if totals else None,
            "decode_reported_max_bytes_range": [min(maxima), max(maxima)] if maxima else None,
            "attribution_limit": "kernel records carry processID=0; actor text and the request time window correlate them, but do not prove PID identity",
        },
        "memory_pressure_events": memory,
        "thermal_events": thermal,
    }


class Recorder:
    def __init__(self, config: Config, *, reader=None, clock=time.time, monotonic=time.monotonic):
        config.validate()
        self.config = config
        self.archive_root = Path(config.archive_root)
        self.pid_file = Path(config.pid_file)
        self.output_root = Path(config.output_root)
        self.service_log = Path(config.service_log)
        self.api_key_file = Path(config.api_key_file)
        ensure_private_directory(self.output_root)
        self.clock = clock
        self.monotonic = monotonic
        self.reader = reader
        self.native_module = None
        self.started_at = clock()
        self.not_before_unix = self.started_at
        self.publication_watermark_ns = int(self.started_at * 1_000_000_000)
        self.publication_watermark_key = "\uffff"
        self._publication_overlap_enabled = False
        self.pid = None
        self.proc_start_abstime = None
        self.target_verification = None
        self.samples = deque(maxlen=max(2, int(config.ring_seconds / config.poll_seconds) + 2))
        self.pending = []
        self.processed: dict[str, int] = {}
        self.archive_scan_degraded = False
        self.archive_scan_degraded_reason = None
        self.last_scan_wall_ns = int(self.started_at * 1_000_000_000)
        self.admission_by_process: dict[str, dict] = {}
        self._state_dirty = False
        now_monotonic = monotonic()
        # Archive scan runs immediately, then at its independent lower rate.
        self._last_archive_scan = now_monotonic - config.archive_scan_seconds
        self._last_state_checkpoint = now_monotonic
        self._last_prune = now_monotonic
        self._limited_events = {}
        self.events_path = self.output_root / "events.jsonl"
        self.state_path = self.output_root / "state.json"
        self._load_state()

    def _load_state(self):
        try:
            state_bytes, _ = read_owned_regular(self.state_path, max_bytes=MAX_STATE_BYTES)
            state = json.loads(state_bytes)
        except (FileNotFoundError, OSError, json.JSONDecodeError, RuntimeError):
            return
        if not isinstance(state, dict):
            return
        value = state.get("not_before_unix")
        if (isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(float(value))):
            self.not_before_unix = float(value)
        publication = state.get("publication_watermark")
        if isinstance(publication, dict):
            mtime_ns = publication.get("mtime_ns")
            key = publication.get("key")
            if (isinstance(mtime_ns, int) and not isinstance(mtime_ns, bool)
                    and mtime_ns >= 0 and isinstance(key, str)):
                self.publication_watermark_ns = mtime_ns
                self.publication_watermark_key = key
                self._publication_overlap_enabled = True
        else:
            # Schema v1/v2 was never deployed.  If a prototype state exists,
            # migrate at its last durable write time rather than mixing the old
            # request-finished watermark with file publication order.
            migrated = state.get("updated_at_unix", self.not_before_unix)
            if (isinstance(migrated, (int, float)) and not isinstance(migrated, bool)
                    and math.isfinite(float(migrated))):
                self.publication_watermark_ns = int(float(migrated) * 1_000_000_000)
                self.publication_watermark_key = "\uffff"
        rows = state.get("processed_requests")
        if isinstance(rows, list):
            for row in rows[-PROCESSED_JOURNAL_LIMIT:]:
                if not isinstance(row, dict) or not valid_processed_key(row.get("key")):
                    continue
                published = row.get("publication_mtime_ns")
                if (isinstance(published, int) and not isinstance(published, bool)
                        and published >= 0):
                    self.processed[row["key"]] = published
        else:
            # Prototype schema v1 compatibility.  Publication watermark, not
            # UUID ordering, prevents omitted legacy entries from replaying.
            for key in state.get("processed_request_ids", [])[-PROCESSED_JOURNAL_LIMIT:]:
                if valid_processed_key(key):
                    self.processed[key] = self.publication_watermark_ns
        self.archive_scan_degraded = state.get("archive_scan_degraded") is True
        degraded_reason = state.get("archive_scan_degraded_reason")
        if isinstance(degraded_reason, str) and degraded_reason:
            self.archive_scan_degraded_reason = degraded_reason
        elif self.archive_scan_degraded:
            self.archive_scan_degraded_reason = "legacy_degraded_state"
        last_scan_wall_ns = state.get("last_scan_wall_ns")
        if (isinstance(last_scan_wall_ns, int) and not isinstance(last_scan_wall_ns, bool)
                and last_scan_wall_ns >= 0):
            self.last_scan_wall_ns = last_scan_wall_ns
        pending = state.get("pending")
        if isinstance(pending, list):
            for item in pending:
                record = item.get("record") if isinstance(item, dict) else None
                ready_at = item.get("ready_at") if isinstance(item, dict) else None
                finished = record.get("finished_unix") if isinstance(record, dict) else None
                duration = (
                    record.get("generation_duration_s") if isinstance(record, dict) else None
                )
                if (isinstance(record, dict)
                        and canonical_uuid(record.get("archive_request_id")) is not None
                        and isinstance(ready_at, (int, float)) and not isinstance(ready_at, bool)
                        and math.isfinite(float(ready_at))
                        and isinstance(finished, (int, float)) and not isinstance(finished, bool)
                        and math.isfinite(float(finished))
                        and isinstance(duration, (int, float)) and not isinstance(duration, bool)
                        and math.isfinite(float(duration)) and float(duration) >= 0):
                    self.pending.append(item)
                    break
        admission = state.get("admission_by_process")
        if isinstance(admission, dict):
            for key, row in admission.items():
                if (not isinstance(key, str) or re.fullmatch(r"[0-9a-f]{64}", key) is None
                        or not isinstance(row, dict)):
                    continue
                last = row.get("last_admitted_unix")
                count = row.get("suppressed_since_last_admission", 0)
                if (isinstance(last, (int, float)) and not isinstance(last, bool)
                        and math.isfinite(float(last)) and isinstance(count, int)
                        and not isinstance(count, bool) and count >= 0):
                    self.admission_by_process[key] = {
                        "last_admitted_unix": float(last),
                        "suppressed_since_last_admission": count,
                    }

    def _prune_processed_index(self) -> None:
        if len(self.processed) <= PROCESSED_JOURNAL_LIMIT:
            return
        cutoff_ns = max(
            int(self.not_before_unix * 1_000_000_000),
            self.publication_watermark_ns - PUBLICATION_OVERLAP_NS,
        )
        newest = sorted(
            self.processed.items(), key=lambda item: (item[1], item[0]), reverse=True
        )
        protected = [item for item in newest if item[1] >= cutoff_ns]
        if len(protected) > PROCESSED_JOURNAL_LIMIT:
            # Exact de-duplication needs every key still inside the overlap.
            # This should be unreachable for a single-model service (10k
            # terminal publications in five seconds), but fail closed rather
            # than evicting a live key and creating a capture storm.
            self.archive_scan_degraded = True
            self.archive_scan_degraded_reason = (
                "publication_overlap_exceeded_processed_journal_capacity"
            )
            self._state_dirty = True
            newest = protected[:PROCESSED_JOURNAL_LIMIT]
        else:
            protected_keys = {key for key, _ in protected}
            safe = [item for item in newest if item[0] not in protected_keys]
            newest = protected + safe[:PROCESSED_JOURNAL_LIMIT - len(protected)]
        self.processed = dict(newest)

    def _mark_processed(
        self, key: str, publication_mtime_ns: int, *, advance_watermark: bool = True
    ) -> None:
        self.processed[key] = publication_mtime_ns
        if advance_watermark:
            candidate = (publication_mtime_ns, key)
            current = (self.publication_watermark_ns, self.publication_watermark_key)
            if candidate > current:
                self.publication_watermark_ns, self.publication_watermark_key = candidate
        self._state_dirty = True

    def _save_state(self):
        self._prune_processed_index()
        processed = sorted(self.processed.items(), key=lambda item: (item[1], item[0]))
        admission = sorted(
            self.admission_by_process.items(),
            key=lambda item: item[1]["last_admitted_unix"], reverse=True,
        )[:16]
        self.admission_by_process = dict(admission)
        atomic_json(self.state_path, {
            "schema_version": 3,
            "updated_at_unix": self.clock(),
            "not_before_unix": self.not_before_unix,
            "publication_watermark": {
                "mtime_ns": self.publication_watermark_ns,
                "key": self.publication_watermark_key,
            },
            "archive_scan_degraded": self.archive_scan_degraded,
            "archive_scan_degraded_reason": self.archive_scan_degraded_reason,
            "last_scan_wall_ns": self.last_scan_wall_ns,
            "pid": self.pid,
            "proc_start_abstime": self.proc_start_abstime,
            "processed_requests": [
                {"key": key, "publication_mtime_ns": published}
                for key, published in processed
            ],
            "pending": self.pending,
            "admission_by_process": self.admission_by_process,
        })
        self._state_dirty = False
        self._last_state_checkpoint = self.monotonic()

    def _checkpoint_state(self, *, force=False):
        due = self.monotonic() - self._last_state_checkpoint >= self.config.state_checkpoint_seconds
        if self._state_dirty and (force or due):
            try:
                self._save_state()
            except Exception as error:
                # Recorder persistence must never take down process/host
                # sampling.  Keep dirty=True so a later checkpoint retries.
                self._state_dirty = True
                try:
                    self._event_limited(
                        "state_checkpoint_error",
                        error_type=type(error).__name__,
                        detail=str(error)[:512],
                    )
                except Exception:
                    pass
                return False
        return True

    def _event(self, kind: str, **values):
        self._rotate_events_if_needed()
        append_private(self.events_path, {"at_unix": self.clock(), "kind": kind, **values})

    def _event_limited(self, kind: str, *, interval_seconds=60.0, **values):
        fingerprint = canonical_digest({"kind": kind, **values})
        now = self.clock()
        previous = self._limited_events.get(kind)
        if previous and previous["fingerprint"] == fingerprint and now - previous["last_written"] < interval_seconds:
            previous["suppressed"] += 1
            return
        suppressed = previous.get("suppressed", 0) if previous else 0
        self._limited_events[kind] = {
            "fingerprint": fingerprint, "last_written": now, "suppressed": 0,
        }
        self._event(kind, suppressed_repeats_since_previous=suppressed, **values)

    def _rotate_events_if_needed(self):
        try:
            if self.events_path.stat().st_size < self.config.max_event_file_bytes:
                return
        except FileNotFoundError:
            return
        for index in range(self.config.event_file_rotations, 0, -1):
            source = self.events_path if index == 1 else self.events_path.with_name(
                self.events_path.name + f".{index - 1}"
            )
            destination = self.events_path.with_name(self.events_path.name + f".{index}")
            if index == self.config.event_file_rotations:
                destination.unlink(missing_ok=True)
            if source.exists() and not source.is_symlink():
                os.replace(source, destination)

    def _resolve_pid(self):
        from control import owns, read_record
        record = read_record()
        if not owns(record):
            raise ProcessLookupError("No owned service process")
        return record["pid"]

    def _verify_target(self, pid: int, *, force=False) -> dict:
        if force or self.target_verification is None or self.target_verification.get("pid") != pid:
            verification = verify_service_identity(pid)
            old_contract = None if self.target_verification is None else {
                key: self.target_verification.get(key) for key in ("pid", "uid", "cwd", "command", "listeners")
            }
            new_contract = {key: verification.get(key) for key in ("pid", "uid", "cwd", "command", "listeners")}
            self.target_verification = verification
            if old_contract != new_contract:
                self._event("target_verified", **verification)
        return self.target_verification

    def sample(self):
        try:
            pid = self._resolve_pid()
            self._verify_target(pid)
            if self.reader is None:
                self.reader, self.native_module = load_native_reader()
            raw = self.reader(pid)
            row = compact_sample(raw, observed_pid=pid)
            row["service_log"] = file_position(self.service_log)
            row["target_identity_verification_sha256"] = canonical_digest(self.target_verification)
            start = row["process"].get("proc_start_abstime")
            if self.pid != pid or self.proc_start_abstime != start:
                self._event("target_identity", old_pid=self.pid, pid=pid, proc_start_abstime=start)
                self.pid, self.proc_start_abstime = pid, start
                self._state_dirty = True
            self.samples.append(row)
            self._spool(row)
        except Exception as error:
            self._event_limited("sample_error", error_type=type(error).__name__, detail=str(error)[:512])

    def _spool(self, row: dict):
        timestamp = datetime.fromtimestamp(row["observed_at_unix"], LOCAL_TZ)
        path = self.output_root / "spool" / timestamp.strftime("%Y%m%d-%H.jsonl")
        append_private(path, row)

    def _candidate_days(self) -> list[Path]:
        today = datetime.fromtimestamp(self.clock(), LOCAL_TZ).date()
        return [self.archive_root / (today - timedelta(days=offset)).isoformat() for offset in (0, 1)]

    def scan_requests(self):
        if self.archive_scan_degraded:
            try:
                self._event_limited(
                    "archive_scan_degraded",
                    reason=(self.archive_scan_degraded_reason or "unknown"),
                    interval_seconds=3600.0,
                )
            except Exception:
                pass
            return
        scan_started_unix = self.clock()
        scan_started_ns = int(scan_started_unix * 1_000_000_000)
        if scan_started_ns + PUBLICATION_OVERLAP_NS < self.last_scan_wall_ns:
            self.archive_scan_degraded = True
            self.archive_scan_degraded_reason = "wall_clock_rollback"
            self._state_dirty = True
            try:
                self._event(
                    "archive_scan_degraded",
                    reason=self.archive_scan_degraded_reason,
                )
            except Exception:
                pass
            # The fail-closed decision must survive an immediate recorder
            # restart; it is too important to wait for the normal checkpoint.
            self._checkpoint_state(force=True)
            return
        settled_horizon_ns = int(
            max(self.not_before_unix, scan_started_unix - ARCHIVE_SETTLE_SECONDS)
            * 1_000_000_000
        )
        current_watermark = (
            self.publication_watermark_ns, self.publication_watermark_key
        )
        proposed_watermark = max(
            current_watermark, (settled_horizon_ns, "\uffff")
        )
        enumeration_had_error = False
        for day in self._candidate_days():
            try:
                directories = [
                    entry for entry in day.iterdir()
                    if entry.is_dir() and not entry.is_symlink()
                ]
            except FileNotFoundError:
                continue
            except OSError as error:
                enumeration_had_error = True
                try:
                    self._event_limited(
                        "archive_scan_error", error_type=type(error).__name__,
                        day_sha256=hashlib.sha256(str(day).encode()).hexdigest(),
                    )
                except Exception:
                    pass
                continue
            for directory in directories:
                canonical_id = canonical_uuid(directory.name)
                lookup_key = canonical_id or (
                    "invalid-name-sha256:"
                    + hashlib.sha256(directory.name.encode(errors="replace")).hexdigest()
                )
                if lookup_key in self.processed:
                    continue
                try:
                    metadata_info = (directory / "metadata.json").lstat()
                    if (stat.S_ISLNK(metadata_info.st_mode)
                            or not stat.S_ISREG(metadata_info.st_mode)
                            or metadata_info.st_uid != os.getuid()
                            or metadata_info.st_nlink != 1
                            or metadata_info.st_size > MAX_METADATA_BYTES):
                        raise RuntimeError("unsafe archive metadata file")
                    publication_mtime_ns = metadata_info.st_mtime_ns
                    # Publication order is the sole durable cursor.  Request
                    # finished_unix is allowed to move backwards under
                    # concurrent completion and never participates here.
                    if publication_mtime_ns <= int(self.not_before_unix * 1_000_000_000):
                        continue
                    overlap = (
                        PUBLICATION_OVERLAP_NS if self._publication_overlap_enabled else 0
                    )
                    if publication_mtime_ns < current_watermark[0] - overlap:
                        continue
                    if (not self._publication_overlap_enabled
                            and (publication_mtime_ns, lookup_key) <= current_watermark):
                        continue
                except FileNotFoundError:
                    continue
                except (OSError, RuntimeError) as error:
                    try:
                        self._event_limited(
                            "archive_entry_error", error_type=type(error).__name__,
                            archive_key_sha256=hashlib.sha256(lookup_key.encode()).hexdigest(),
                        )
                    except Exception:
                        pass
                    continue
                try:
                    outcome = inspect_archive_entry(directory, now=scan_started_unix)
                except Exception as error:
                    try:
                        self._event_limited(
                            "archive_entry_error", error_type=type(error).__name__,
                            archive_key_sha256=hashlib.sha256(lookup_key.encode()).hexdigest(),
                        )
                    except Exception:
                        pass
                    continue
                if outcome["state"] == "incomplete":
                    continue
                if outcome["state"] == "error":
                    try:
                        self._event_limited(
                            "archive_entry_error", error_type=outcome["reason"],
                            archive_key_sha256=hashlib.sha256(lookup_key.encode()).hexdigest(),
                        )
                    except Exception:
                        pass
                    continue
                publication_mtime_ns = int(outcome["publication_mtime_ns"])
                self._mark_processed(
                    outcome["key"], publication_mtime_ns, advance_watermark=False
                )
                if outcome["state"] == "terminal_invalid":
                    try:
                        self._event_limited(
                            "terminal_archive_invalid",
                            reason=outcome["reason"],
                            archive_key_sha256=hashlib.sha256(outcome["key"].encode()).hexdigest(),
                        )
                    except Exception:
                        pass
                    continue
                record = outcome["record"]
                request_id = record["archive_request_id"]
                reasons = trigger_reasons(record, self.config)
                if reasons:
                    self._admit_trigger(record, reasons)
                try:
                    self._event(
                        "request_completed",
                        request_id=request_id,
                        engine_request_ids=record["engine_request_ids"],
                        completion_tokens=record["completion_tokens"],
                        generation_tokens_per_second=record["generation_tokens_per_second"],
                        trigger_reasons=reasons,
                    )
                except Exception:
                    pass
        if not enumeration_had_error and proposed_watermark > current_watermark:
            self.publication_watermark_ns, self.publication_watermark_key = proposed_watermark
            self._state_dirty = True
        if not enumeration_had_error:
            if scan_started_ns > self.last_scan_wall_ns:
                self.last_scan_wall_ns = scan_started_ns
                self._state_dirty = True
            self._publication_overlap_enabled = True
        self._prune_processed_index()

    def _admit_trigger(self, record: dict, reasons: list[str]) -> None:
        process_key = canonical_digest({
            "pid": self.pid, "proc_start_abstime": self.proc_start_abstime,
        })
        now = self.clock()
        admission = self.admission_by_process.setdefault(process_key, {
            "last_admitted_unix": 0.0,
            "suppressed_since_last_admission": 0,
        })
        if len(self.pending) >= self.config.max_pending_incidents:
            pending_admission = self.pending[0].setdefault("admission", {})
            pending_admission["suppressed_while_pending"] = (
                int(pending_admission.get("suppressed_while_pending", 0)) + 1
            )
            admission["suppressed_since_last_admission"] += 1
            self._state_dirty = True
            try:
                self._event_limited(
                    "trigger_suppressed", reason="pending_cap",
                    process_identity_sha256=process_key,
                )
            except Exception:
                pass
            return
        elapsed = now - admission["last_admitted_unix"]
        if elapsed < self.config.trigger_cooldown_seconds:
            admission["suppressed_since_last_admission"] += 1
            self._state_dirty = True
            try:
                self._event_limited(
                    "trigger_suppressed", reason="process_cooldown",
                    process_identity_sha256=process_key,
                )
            except Exception:
                pass
            return
        suppressed_before = admission["suppressed_since_last_admission"]
        admission["last_admitted_unix"] = now
        admission["suppressed_since_last_admission"] = 0
        self.pending.append({
            "ready_at": max(now, record["finished_unix"] + self.config.postroll_seconds),
            "record": record,
            "reasons": reasons,
            "pid": self.pid,
            "proc_start_abstime": self.proc_start_abstime,
            "target_verification": self.target_verification,
            "attempts": 0,
            "admission": {
                "process_identity_sha256": process_key,
                "cooldown_seconds": self.config.trigger_cooldown_seconds,
                "suppressed_since_previous_admission": suppressed_before,
                "suppressed_while_pending": 0,
            },
        })
        self._state_dirty = True

    def finalize_ready(self):
        before_count = len(self.pending)
        remaining = []
        for item in self.pending:
            if self.clock() < item["ready_at"]:
                remaining.append(item)
                continue
            try:
                self._finalize(item)
            except Exception as error:
                item["attempts"] = int(item.get("attempts", 0)) + 1
                item["last_error"] = {"type": type(error).__name__, "detail": str(error)[:512]}
                try:
                    self._event_limited(
                        "incident_capture_error",
                        request_id=item["record"]["archive_request_id"],
                        error_type=type(error).__name__,
                        detail=str(error)[:512],
                    )
                except Exception:
                    pass
                if item["attempts"] < self.config.finalize_retries:
                    item["ready_at"] = self.clock() + 30
                    remaining.append(item)
                    self._state_dirty = True
                else:
                    try:
                        self._seal_failed_incident(item)
                    except Exception as seal_error:
                        item["ready_at"] = self.clock() + 60
                        item["last_seal_error"] = {
                            "type": type(seal_error).__name__,
                            "detail": str(seal_error)[:512],
                        }
                        remaining.append(item)
                        self._state_dirty = True
        self.pending = remaining
        if len(remaining) != before_count:
            self._state_dirty = True
        self._checkpoint_state(force=True)

    @staticmethod
    def _incident_id(record: dict) -> str:
        request_id = canonical_uuid(record.get("archive_request_id"))
        finished = record.get("finished_unix")
        if request_id is None:
            raise ValueError("incident archive_request_id is not a canonical UUID")
        if (not isinstance(finished, (int, float)) or isinstance(finished, bool)
                or not math.isfinite(float(finished))):
            raise ValueError("incident finished_unix is invalid")
        return (
            datetime.fromtimestamp(float(finished), LOCAL_TZ).strftime("%Y%m%d-%H%M%S")
            + "-" + uuid.UUID(request_id).hex
        )

    def _incidents_root(self) -> Path:
        root = self.output_root / "incidents"
        ensure_private_directory(root)
        output_resolved = self.output_root.resolve(strict=True)
        if root.resolve(strict=True).parent != output_resolved or root.is_symlink():
            raise RuntimeError("incidents root escaped the private output root")
        return root

    def _incident_path(self, name: str) -> Path:
        if not INCIDENT_NAME_RE.fullmatch(name):
            raise ValueError("incident path component does not match the internal schema")
        root = self._incidents_root()
        candidate = root / name
        if candidate.parent.resolve(strict=True) != root.resolve(strict=True):
            raise RuntimeError("incident path escaped its root")
        if candidate.is_symlink():
            raise RuntimeError("incident path is a symlink")
        if candidate.exists() and candidate.resolve(strict=True).parent != root.resolve(strict=True):
            raise RuntimeError("existing incident path escaped its root")
        return candidate

    def _remove_incident_directory(self, path: Path, *, staging_only: bool = False) -> None:
        root = self._incidents_root().resolve(strict=True)
        if staging_only and not path.name.startswith(".partial-"):
            raise RuntimeError("refusing to remove a non-staging incident")
        if not INCIDENT_NAME_RE.fullmatch(path.name):
            raise RuntimeError("refusing to remove an unknown incident directory")
        info = path.lstat()
        if (stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode)
                or info.st_uid != os.getuid() or path.resolve(strict=True).parent != root):
            raise RuntimeError("refusing to remove an unsafe incident directory")
        for child in path.iterdir():
            child_info = child.lstat()
            if (stat.S_ISLNK(child_info.st_mode) or not stat.S_ISREG(child_info.st_mode)
                    or child_info.st_uid != os.getuid() or child_info.st_nlink != 1):
                raise RuntimeError("refusing to remove unsafe incident content")
            child.unlink()
        path.rmdir()

    def _seal_failed_incident(self, item: dict):
        incident_id = self._incident_id(item["record"])
        partial = self._incident_path(".partial-" + incident_id)
        if partial.exists():
            self._remove_incident_directory(partial, staging_only=True)
        ensure_private_directory(partial)
        atomic_json(partial / "capture-failure.json", {
            "schema_version": 1,
            "status": "capture_failed",
            "attempts": item.get("attempts"),
            "last_error": item.get("last_error"),
            "request": item.get("record"),
            "sealed_at_unix": self.clock(),
        })
        destination = self._incident_path(incident_id + "-failed")
        if destination.exists():
            return
        os.replace(partial, destination)

    def _finalize(self, item: dict):
        record = item["record"]
        decode_end = float(record["finished_unix"])
        decode_start = decode_end - float(record["generation_duration_s"])
        window_start = decode_start - self.config.log_preroll_seconds
        window_end = decode_end + self.config.postroll_seconds
        incident_id = self._incident_id(record)
        incident = self._incident_path(incident_id)
        if (incident / "manifest.json").exists():
            return
        if incident.exists():
            raise RuntimeError("incident destination exists without a manifest")
        work = self._incident_path(".partial-" + incident_id)
        if work.exists():
            self._remove_incident_directory(work, staging_only=True)
        ensure_private_directory(work)
        samples = [row for row in self.samples
                   if window_start <= row["observed_at_unix"] <= window_end]
        sample_path = work / "os-samples.jsonl"
        fd = os.open(
            sample_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
        )
        verify_private_regular_fd(fd, label="incident sample file")
        os.close(fd)
        for row in samples:
            append_private(sample_path, row)
        manifest = {
            "schema_version": 2,
            "status": "captured_minimal",
            "read_only": True,
            "created_at_unix": self.clock(),
            "incident_id": incident_id,
            "trigger_reasons": item["reasons"],
            "admission": item.get("admission", {}),
            "target": {
                "pid_at_trigger": item["pid"],
                "proc_start_abstime_at_trigger": item["proc_start_abstime"],
                "pid_at_capture": self.pid,
                "proc_start_abstime_at_capture": self.proc_start_abstime,
                "pid_identity_stable": (
                    item["pid"] == self.pid
                    and item["proc_start_abstime"] == self.proc_start_abstime
                ),
                "pid_file": str(self.pid_file),
                "verification_at_trigger": item.get("target_verification"),
            },
            "request": record,
            "windows": {
                "request": [record["started_unix"], record["finished_unix"]],
                "estimated_decode": [decode_start, decode_end],
                "decode_boundary_method": "archive finished_unix minus server generation_duration; no extra target synchronization",
                "decode_duration_rounding_uncertainty_seconds": 0.01,
                "transport_boundary_limit": "archive completion may trail engine completion by response transport/queue latency",
            },
            "os_sampling": {
                "interval_seconds": self.config.poll_seconds,
                "sample_count": len(samples),
                "path": str(sample_path),
                "scope": "process rusage plus host-wide VM/swap/thermal; sampled sequentially",
            },
            "incident_enrichment": {
                "enabled": False,
                "reason": "disabled_by_default_and_not_production_qualified",
                "unified_log": False,
                "runtime_http": False,
                "pmset": False,
                "service_log_slice": False,
                "target_subprocess_snapshots": False,
            },
            "privacy": {
                "archive_request_or_response_payload_copied": False,
                "request_and_response_referenced_by_hash_only": True,
                "client_request_id_stored_as_sha256_only": True,
                "service_log_slice_enabled": False,
            },
            "provenance": {
                "recorder": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__))},
                "native_reader": {"path": str(NATIVE_READER), "sha256": sha256(NATIVE_READER)},
                "config": asdict(self.config),
            },
        }
        manifest = relocate_paths(manifest, work, incident)
        atomic_json(work / "manifest.json", manifest)
        os.replace(work, incident)
        try:
            self._event(
                "incident_captured", request_id=record["archive_request_id"],
                incident=str(incident),
            )
        except Exception:
            pass
        self._state_dirty = True

    def _classify_request_overlap(self, record: dict) -> dict:
        target_start = record["started_unix"]
        target_end = record["finished_unix"]
        dates = {
            datetime.fromtimestamp(target_start, LOCAL_TZ).date(),
            datetime.fromtimestamp(target_end, LOCAL_TZ).date(),
        }
        overlaps = []
        inspected = 0
        errors = 0
        for date in sorted(dates):
            day = self.archive_root / date.isoformat()
            try:
                directories = list(day.iterdir())
            except FileNotFoundError:
                continue
            for directory in directories:
                if not directory.is_dir() or directory.name == record["archive_request_id"]:
                    continue
                try:
                    metadata = json.loads((directory / "metadata.json").read_text())
                    inspected += 1
                    started = metadata.get("started_unix")
                    finished = metadata.get("finished_unix")
                    if not isinstance(started, (int, float)) or isinstance(started, bool):
                        continue
                    effective_end = finished if isinstance(finished, (int, float)) and not isinstance(finished, bool) else self.clock()
                    if float(started) < target_end and float(effective_end) > target_start:
                        overlaps.append({
                            "archive_request_id": metadata.get("request_id", directory.name),
                            "engine_request_ids": list(metadata.get("engine_request_ids") or []),
                            "status": metadata.get("status"),
                            "started_unix": float(started),
                            "finished_unix": float(effective_end),
                        })
                except (OSError, json.JSONDecodeError, TypeError, ValueError):
                    errors += 1
        return {
            "archive_intervals_inspected": inspected,
            "archive_metadata_errors": errors,
            "overlapping_archive_request_count": len(overlaps),
            "overlapping_archive_requests": overlaps[:100],
            "overlap_rows_truncated": len(overlaps) > 100,
            "transport_single_request_supported": not overlaps and errors == 0,
            "scope_limit": "HTTP archive interval overlap is not exact scheduler/GPU overlap; post-request scheduler snapshots cannot reconstruct past concurrency",
        }

    def _capture_service_log_slice(self, incident: Path, samples: list[dict]) -> dict:
        positions = [row.get("service_log") or {} for row in samples]
        positions = [row for row in positions if row.get("available")]
        current = file_position(self.service_log)
        if not positions or not current.get("available"):
            return {"captured": False, "reason": "log_position_unavailable"}
        identity = (current["device"], current["inode"])
        matching = [row for row in positions if (row.get("device"), row.get("inode")) == identity]
        if not matching:
            return {"captured": False, "reason": "service_log_rotated_during_window"}
        start = min(row["size"] for row in matching)
        end_observed = current["size"]
        requested = max(0, end_observed - start)
        length = min(requested, self.config.max_service_log_slice_bytes)
        output = incident / "service-log-slice.txt"
        fd = os.open(self.service_log, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            data = os.pread(fd, length, start)
            after = os.fstat(fd)
        finally:
            os.close(fd)
        if (after.st_dev, after.st_ino) != identity:
            return {"captured": False, "reason": "service_log_identity_changed_during_read"}
        out_fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(out_fd, "wb") as stream:
            stream.write(data)
        return {
            "captured": True,
            "path": str(output),
            "device": identity[0],
            "inode": identity[1],
            "start_offset": start,
            "end_offset_observed": end_observed,
            "bytes_written": len(data),
            "truncated": requested > len(data),
            "scope": "bounded byte slice anchored by service-log offsets in OS samples; log lines have no wall-clock prefix",
        }

    def _authenticated_json(self, path: str) -> dict:
        if path not in {"/api/runtime-profile", "/api/v41-prefix-cache"}:
            raise ValueError("snapshot endpoint is not allowlisted")
        key_bytes, _ = read_owned_regular(self.api_key_file, max_bytes=MAX_API_KEY_BYTES)
        key = key_bytes.decode("utf-8").strip()
        if not key:
            raise RuntimeError("empty API key file")
        url = self.config.base_url + path
        request = urllib.request.Request(
            url,
            headers={"Authorization": "Bearer " + key},
            method="GET",
        )
        # Do not inherit HTTP(S)_PROXY or system proxy settings.  This also
        # keeps the bearer token pinned to the validated loopback endpoint.
        opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}), NoRedirectHandler()
        )
        with opener.open(request, timeout=15) as response:
            if response.geturl() != url:
                raise RuntimeError("snapshot endpoint redirected away from its pinned URL")
            payload = response.read(self.config.max_snapshot_bytes + 1)
        if len(payload) > self.config.max_snapshot_bytes:
            raise RuntimeError("snapshot response exceeded its byte bound")
        result = json.loads(payload)
        if not isinstance(result, dict):
            raise RuntimeError("snapshot response was not a JSON object")
        return result

    def _write_bounded_command(self, path: Path, command: list[str], *, timeout: int = 60) -> dict:
        completed = subprocess.run(command, capture_output=True, timeout=timeout, check=False)
        data = completed.stdout[: self.config.max_snapshot_bytes]
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
        return {
            "path": str(path),
            "command": command,
            "returncode": completed.returncode,
            "stdout_bytes": len(completed.stdout),
            "stdout_truncated": len(completed.stdout) > len(data),
            "stderr": completed.stderr.decode(errors="replace")[:4096],
        }

    def _capture_post_request_snapshots(
        self, incident: Path, expected_pid: int | None, expected_start: int | None
    ) -> dict:
        result = {"trigger_only": True, "api_key_stored": False, "target_tools_started": False}
        # pmset is host-only and does not inspect or synchronize the target.
        try:
            result["pmset_therm"] = self._write_bounded_command(
                incident / "pmset-therm.txt", ["/usr/bin/pmset", "-g", "therm"], timeout=15
            )
        except Exception as error:
            result["pmset_therm"] = {"error_type": type(error).__name__, "detail": str(error)[:512]}
        if not isinstance(expected_pid, int) or not isinstance(expected_start, int):
            result["target_snapshot_skip_reasons"] = ["trigger_pid_identity_unavailable"]
            return result
        try:
            identity_contract = verify_service_identity(expected_pid)
            result["target_identity_contract"] = identity_contract
            profile_before = self._authenticated_json("/api/runtime-profile")
            cache_before = self._authenticated_json("/api/v41-prefix-cache")
            atomic_json(incident / "runtime-profile-before.json", profile_before)
            atomic_json(incident / "prefix-cache-before.json", cache_before)
            result["runtime_profile_before_sha256"] = canonical_digest(profile_before)
            result["prefix_cache_before_sha256"] = canonical_digest(cache_before)
            idle, reasons = runtime_is_idle(profile_before, cache_before, expected_pid)
            identity = compact_sample(self.reader(expected_pid), observed_pid=expected_pid)
            if identity["process"].get("proc_start_abstime") != expected_start:
                reasons.append("proc_start_abstime_changed")
                idle = False
            if not idle:
                result["target_snapshot_skip_reasons"] = reasons
                return result
            if self.config.enable_target_subprocess_snapshots:
                result["target_tools_started"] = True
                footprint_path = incident / "footprint.json"
                footprint = subprocess.run(
                    ["/usr/bin/footprint", "--noCategories", "--swapped", "--wired",
                     "--json", str(footprint_path), "--pid", str(expected_pid)],
                    capture_output=True, timeout=60, check=False,
                )
                try:
                    footprint_size = footprint_path.stat().st_size
                except OSError:
                    footprint_size = 0
                if footprint_size > self.config.max_snapshot_bytes:
                    footprint_path.unlink(missing_ok=True)
                    footprint_too_large = True
                else:
                    footprint_too_large = False
                result["footprint"] = {
                    "captured": footprint.returncode == 0 and not footprint_too_large,
                    "path": str(footprint_path), "returncode": footprint.returncode,
                    "bytes": footprint_size, "removed_for_size_bound": footprint_too_large,
                    "stdout": footprint.stdout.decode(errors="replace")[:4096],
                    "stderr": footprint.stderr.decode(errors="replace")[:4096],
                }
            else:
                result["footprint"] = {
                    "captured": False,
                    "reason": "disabled_by_default_enable_target_subprocess_snapshots_required",
                }
            result["vmmap_summary"] = {
                "captured": False,
                "reason": "manual_maintenance_only_after_measured_10.846_second_idle_snapshot",
            }
            profile_after = self._authenticated_json("/api/runtime-profile")
            cache_after = self._authenticated_json("/api/v41-prefix-cache")
            atomic_json(incident / "runtime-profile-after.json", profile_after)
            atomic_json(incident / "prefix-cache-after.json", cache_after)
            result["runtime_profile_after_sha256"] = canonical_digest(profile_after)
            result["prefix_cache_after_sha256"] = canonical_digest(cache_after)
            idle_after, after_reasons = runtime_is_idle(profile_after, cache_after, expected_pid)
            identity_after = compact_sample(self.reader(expected_pid), observed_pid=expected_pid)
            if identity_after["process"].get("proc_start_abstime") != expected_start:
                after_reasons.append("proc_start_abstime_changed_after_snapshots")
                idle_after = False
            result["idle_and_identity_stable_after_snapshots"] = idle_after
            result["post_snapshot_validation_reasons"] = after_reasons
        except Exception as error:
            result["error_type"] = type(error).__name__
            result["detail"] = str(error)[:512]
        return result

    def prune(self):
        now = self.clock()
        spool_cutoff = now - self.config.spool_hours * 3600
        spool_root = self.output_root / "spool"
        if spool_root.exists() or spool_root.is_symlink():
            ensure_private_directory(spool_root)
            for path in spool_root.glob("*.jsonl"):
                try:
                    info = path.lstat()
                    if (stat.S_ISREG(info.st_mode) and info.st_uid == os.getuid()
                            and info.st_nlink == 1 and info.st_mtime < spool_cutoff):
                        path.unlink()
                except OSError:
                    continue
        root = self._incidents_root()
        incidents = []
        for path in root.iterdir():
            try:
                info = path.lstat()
                if (not INCIDENT_NAME_RE.fullmatch(path.name)
                        or stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode)
                        or info.st_uid != os.getuid()
                        or path.resolve(strict=True).parent != root.resolve(strict=True)):
                    continue
                incidents.append((path, info.st_mtime))
            except OSError:
                continue
        incidents.sort(key=lambda item: item[1], reverse=True)
        cutoff = now - self.config.incident_days * 86400
        for index, (path, mtime) in enumerate(incidents):
            try:
                expired = mtime < cutoff or index >= self.config.max_incidents
                if expired:
                    self._remove_incident_directory(path)
            except (OSError, RuntimeError):
                continue

    def tick(self):
        self.sample()
        now = self.monotonic()
        if now - self._last_archive_scan >= self.config.archive_scan_seconds:
            try:
                if self.pid is not None:
                    self._verify_target(self.pid, force=True)
            except Exception as error:
                self.target_verification = None
                self._event_limited(
                    "target_verification_error",
                    error_type=type(error).__name__, detail=str(error)[:512],
                )
            self.scan_requests()
            self._last_archive_scan = now
        self.finalize_ready()
        if now - self._last_prune >= self.config.prune_seconds:
            self.prune()
            self._last_prune = now
        self._checkpoint_state()

    def run(self, *, stop_event: threading.Event | None = None):
        lock_fd = self._acquire_singleton()
        stop_event = stop_event or threading.Event()
        stop_reason = {"value": "external_stop_event" if stop_event.is_set() else "loop_exit"}
        previous_handlers = {}

        def request_stop(signum, _frame):
            # Keep the signal handler async-safe at the Python level: it only
            # updates memory.  Writes happen below after an in-progress tick.
            stop_reason["value"] = signal.Signals(signum).name.lower()
            stop_event.set()

        try:
            for signum in (signal.SIGTERM, signal.SIGINT):
                previous_handlers[signum] = signal.signal(signum, request_stop)
            self._event("recorder_started", config=asdict(self.config), controller_pid=os.getpid())
            while not stop_event.is_set():
                started = time.monotonic()
                self.tick()
                stop_event.wait(max(0, self.config.poll_seconds - (time.monotonic() - started)))
        except KeyboardInterrupt:
            stop_reason["value"] = "keyboard_interrupt"
        finally:
            try:
                self._event("recorder_stopped", reason=stop_reason["value"])
            finally:
                try:
                    self._checkpoint_state(force=True)
                finally:
                    for signum, handler in previous_handlers.items():
                        signal.signal(signum, handler)
                    os.close(lock_fd)

    def _acquire_singleton(self) -> int:
        path = self.output_root / "recorder.lock"
        fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        try:
            info = os.fstat(fd)
            if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                    or info.st_nlink != 1):
                raise RuntimeError("recorder lock must be an owned regular file")
            os.fchmod(fd, 0o600)
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            os.ftruncate(fd, 0)
            os.write(fd, (str(os.getpid()) + "\n").encode())
            return fd
        except BaseException:
            os.close(fd)
            raise
