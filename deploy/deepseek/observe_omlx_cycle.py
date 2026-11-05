#!/usr/bin/env python3
"""Observe a manual oMLX lifecycle with metadata only; never control a service.

Stop this observer with SIGTERM/SIGINT or its stop-marker file. The final
sample and summary are written before exit. No model, MLX, inference API,
service start/stop, task-control port, or VM payload is used.
"""
from __future__ import annotations

import argparse
import ctypes as C
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

from control import owns, read_record

def running():
    record = read_record()
    return record["pid"] if owns(record) else None

from omlx_task_memory_observer import Metadata, GLOBAL_ACTIVITY, delta, task_deltas
from observe_macos_memory import reader as process_rusage_reader
from v41_load_guard import MacTelemetry

GIB = 2**30
DEFAULT_ALLOWED_SWAP_BYTES = 100 * 2**20
MAX_ALLOWED_COMPRESSION_BYTES = GIB
MAX_ALLOWED_COMPRESSOR_GROWTH_BYTES = 2 * GIB
STOP = False


class SwapUsage(C.Structure):
    _fields_ = [("total", C.c_uint64), ("available", C.c_uint64),
                ("used", C.c_uint64), ("page_size", C.c_uint32),
                ("encrypted", C.c_int32)]


def swap_usage(provider):
    result = SwapUsage()
    size = C.c_size_t(C.sizeof(result))
    if provider.lib.sysctlbyname(b"vm.swapusage", C.byref(result), C.byref(size), None, 0):
        raise RuntimeError("Cannot read existing swap usage")
    if size.value != C.sizeof(result):
        raise RuntimeError("Unexpected swap usage ABI")
    return {"total_bytes": result.total, "available_bytes": result.available,
            "used_bytes": result.used, "page_size": result.page_size,
            "encrypted": bool(result.encrypted)}


def process_summary(samples):
    groups = {}
    for row in samples:
        service = row.get("service")
        if service is None:
            continue
        key = service["start_id"]
        group = groups.setdefault(key, {"pid": service["pid"], "start_id": key,
                                       "first": service, "last": service, "samples": 0,
                                       "first_elapsed_s": row["elapsed_s"], "peak_phys_bytes": 0,
                                       "peak_lifetime_phys_bytes": 0})
        group["last"] = service
        group["samples"] += 1
        group["last_elapsed_s"] = row["elapsed_s"]
        group["peak_phys_bytes"] = max(group["peak_phys_bytes"], service["phys_footprint_bytes"])
        group["peak_lifetime_phys_bytes"] = max(group["peak_lifetime_phys_bytes"], service.get("rusage_lifetime_max_phys_bytes", 0))
    for group in groups.values():
        group["observed_ledger_delta"] = delta(group["last"], group["first"],
            ("compressed_lifetime_bytes", "decompressions", "swapin_bytes"))
    return list(groups.values())


def summarize(samples, errors, *, allowed_swap_bytes=DEFAULT_ALLOWED_SWAP_BYTES,
              allowed_compression_bytes=0, allowed_compressor_growth_bytes=None,
              expect_final_stopped=False):
    if type(allowed_swap_bytes) is not int or not 0 <= allowed_swap_bytes <= DEFAULT_ALLOWED_SWAP_BYTES:
        raise ValueError("Swap allowance must be an integer from 0 to 104857600 bytes")
    if type(allowed_compression_bytes) is not int or not 0 <= allowed_compression_bytes <= MAX_ALLOWED_COMPRESSION_BYTES:
        raise ValueError("Compression allowance must be an integer from 0 to 1073741824 bytes")
    if allowed_compressor_growth_bytes is not None:
        if type(allowed_compressor_growth_bytes) is not int or not 0 <= allowed_compressor_growth_bytes <= MAX_ALLOWED_COMPRESSOR_GROWTH_BYTES:
            raise ValueError("Compressor growth allowance must be an integer from 0 to 2147483648 bytes")
        if allowed_compression_bytes != 0:
            raise ValueError("Select physical compressor growth or cumulative compression input, not both")
    findings = list(errors)
    if not samples:
        return {"status": "failed", "errors": [*findings, "No samples recorded"]}
    if len(samples) < 2:
        findings.append("At least two samples are required to measure cumulative paging")
    first, last = samples[0], samples[-1]
    activity = {key: last["activity"][key] - first["activity"][key] for key in GLOBAL_ACTIVITY}
    transitions = [{key: after["activity"][key] - before["activity"][key] for key in GLOBAL_ACTIVITY}
                   for before, after in zip(samples, samples[1:])]
    if any(value < 0 for change in transitions for value in change.values()):
        findings.append("Global cumulative counters decreased during observation")
    page_size = first["activity"]["page_size"]
    valid_pages = (type(page_size) is int and 4096 <= page_size <= 65536
                   and page_size & (page_size - 1) == 0
                   and all(row["activity"].get("page_size") == page_size for row in samples))
    if not valid_pages:
        findings.append("Invalid or changing memory page size")
    swapin_bytes = activity["swapins"] * page_size if valid_pages else None
    swapout_bytes = activity["swapouts"] * page_size if valid_pages else None
    swap_bytes = swapin_bytes + swapout_bytes if valid_pages else None
    # Count cumulative input pages compressed, not net compressor residency.
    # A later decompression never refunds this allowance. Counter resets are
    # independently rejected above; positive transitions cannot hide them.
    compression_bytes = (sum(max(0, change["compressions"]) for change in transitions) * page_size
                         if valid_pages else None)
    strict_zero_compression = all(change["compressions"] == 0 for change in transitions)
    compressor_samples = [row["activity"].get("compressor_physical_bytes") for row in samples]
    compressor_available = all(type(value) is int and value >= 0 for value in compressor_samples)
    compressor_delta = compressor_samples[-1] - compressor_samples[0] if compressor_available else None
    compressor_peak_growth = max(0, max(compressor_samples) - compressor_samples[0]) if compressor_available else None
    raw_zero_gate = all(change[key] == 0 for change in transitions
                        for key in ("compressions", "swapins", "swapouts"))
    if allowed_compressor_growth_bytes is not None:
        if not compressor_available:
            findings.append("Missing or invalid compressor physical telemetry")
        elif compressor_peak_growth > allowed_compressor_growth_bytes:
            findings.append("Compressor physical peak growth exceeded the allowance")
    elif compression_bytes is None or compression_bytes > allowed_compression_bytes:
        findings.append("New system compression observed" if allowed_compression_bytes == 0
                        else "Cumulative new system compression exceeded the allowance")
    if swap_bytes is None or swap_bytes > allowed_swap_bytes:
        findings.append("Cumulative system swap-in plus swap-out exceeded the allowance")
    if any(row["system"]["pressure"] != 1 for row in samples):
        findings.append("System pressure was not normal throughout observation")
    try:
        processes = process_summary(samples)
    except ValueError as error:
        processes = []
        findings.append("Service ledger counters decreased: " + str(error))
    if not processes:
        findings.append("No owned service process was observed")
    for row in samples:
        service = row.get("service")
        if service is not None and service["phys_footprint_bytes"] > min(240 * GIB, row["system"]["total"] - 16 * GIB):
            findings.append("Owned process footprint exceeded the physical/process cap")
            break
    if expect_final_stopped and last.get("pid") is not None:
        findings.append("Service was still running at the final sample")
    latest = next((row for row in reversed(samples) if row.get("service") is not None), None)
    return {"schema_version": 2, "status": "passed" if not findings else "failed",
            "metadata_only": True, "observer_pid": os.getpid(),
            "errors": findings, "sample_count": len(samples),
            "baseline": first, "final": last, "activity_delta": activity,
            "swapin_delta_bytes": swapin_bytes,
            "swapout_delta_bytes": swapout_bytes,
            "swap_delta_bytes": swap_bytes,
            "strict_zero_paging_gate_passed": raw_zero_gate,
            "allowed_swap_bytes": allowed_swap_bytes,
            "compression_delta_bytes": compression_bytes,
            "compression_acceptance_metric": ("physical_peak_growth" if allowed_compressor_growth_bytes is not None
                                              else "cumulative_input"),
            "allowed_compression_bytes": (allowed_compression_bytes if allowed_compressor_growth_bytes is None else None),
            "allowed_compressor_growth_bytes": allowed_compressor_growth_bytes,
            "strict_zero_compression_gate_passed": strict_zero_compression,
            "compressor_physical_delta_bytes": compressor_delta,
            "compressor_physical_peak_growth_bytes": compressor_peak_growth,
            "acceptance_policy": (
                "Cumulative new system swap-in plus swap-out must not exceed its allowance (default 100 MiB). "
                + ("Compressor physical occupancy peak growth above baseline must not exceed the selected allowance; "
                   "a later decrease does not erase the earlier peak. Cumulative newly compressed input bytes are diagnostic only. "
                   if allowed_compressor_growth_bytes is not None else
                   "Cumulative newly compressed input pages must not exceed the separately selected compression allowance "
                   "(default zero, opt-in up to 1 GiB); decompression does not refund it. Compressor physical residency is reported separately. ")
                + "Existing compressor and stored swap are baseline only. Normal pressure, counter validity and physical footprint "
                  "checks remain enforced. Strict-zero status is diagnostic only."),
            "processes": processes,
            "peak_sampled_process_phys_bytes": max((p["peak_phys_bytes"] for p in processes), default=0),
            "minimum_available_bytes": min(row["system"]["available"] for row in samples),
            "total_system_budget_bytes": 16 * GIB, "process_cap_bytes": 240 * GIB,
            "additional_free_reserve_bytes": 0,
            "final_service_stopped": last.get("pid") is None,
            "last_live_process_sample": latest,
            "available_increase_since_last_live_bytes": (last["system"]["available"] - latest["system"]["available"]) if latest else None,
            "existing_compressor_and_stored_swap_are_baseline_not_new_paging": True,
            "between_sample_peaks_unverified": True,
            "lifetime_peak_note": "Per-PID libproc lifetime maxima are also recorded. The first pre-existing service's lifetime peak can include activity before this observer began.",
            "attribution_limit": "Global counter changes are not service attribution. Task deltas identify same-window candidates; inaccessible or exited tasks remain unattributed."}


def recheck_summary(summary_path, *, allowed_swap_bytes=DEFAULT_ALLOWED_SWAP_BYTES,
                    allowed_compression_bytes=0, allowed_compressor_growth_bytes=None,
                    expect_final_stopped=False):
    """Recompute unchanged samples and retain failures unrelated to the allowance."""
    summary_path = Path(summary_path).resolve()
    original_payload = summary_path.read_bytes()
    original = json.loads(original_payload)
    trace_path = Path(original["trace_path"])
    payload = trace_path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != original["trace_sha256"]:
        raise ValueError("Original memory trace digest changed")
    entries = [json.loads(line) for line in payload.decode().splitlines()]
    samples = [row for row in entries if "error" not in row]
    if (not samples or len(samples) != original["sample_count"]
            or samples[0] != original["baseline"] or samples[-1] != original["final"]):
        raise ValueError("Original memory summary does not match its complete trace")
    superseded = {"New system swap-out observed",
                  "System swap-ins exceeded the explicitly selected allowance",
                  "Cumulative system swap-in plus swap-out exceeded the allowance",
                  "New system compression observed",
                  "Cumulative new system compression exceeded the allowance",
                  "Compressor physical peak growth exceeded the allowance",
                  "Missing or invalid compressor physical telemetry"}
    errors = [error for error in original["errors"] if error not in superseded]
    errors.extend(row["error"] for row in entries if "error" in row)
    result = summarize(samples, errors, allowed_swap_bytes=allowed_swap_bytes,
                       allowed_compression_bytes=allowed_compression_bytes,
                       allowed_compressor_growth_bytes=allowed_compressor_growth_bytes,
                       expect_final_stopped=expect_final_stopped)
    result.update(trace_path=str(trace_path), trace_sha256=digest,
                  interval_seconds=original.get("interval_seconds"),
                  reclassification={"original_summary_path": str(summary_path),
                    "original_summary_sha256": hashlib.sha256(original_payload).hexdigest(),
                    "original_status": original["status"], "original_evidence_preserved": True,
                    "policy_authorization": "Owner requested a 100 MiB full-cycle paging allowance on 2026-11-03 and confirmed compressor physical occupancy peak growth of at most 2 GiB on 2026-11-05; the selected metric and allowances are recorded in this result."})
    return result


def stop_signal(signum, frame):
    global STOP
    STOP = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--recheck-summary", type=Path,
                        help="Reassess saved evidence without sampling or contacting the service")
    parser.add_argument("--interval", type=float, default=1)
    parser.add_argument("--duration", type=float, default=7200)
    parser.add_argument("--phase-file", type=Path)
    parser.add_argument("--allowed-global-swap-bytes", type=int, default=DEFAULT_ALLOWED_SWAP_BYTES,
                        help="Cumulative swap-in plus swap-out budget; default 100 MiB")
    parser.add_argument("--allowed-global-compression-bytes", type=int, default=0,
                        help="Cumulative newly compressed input bytes; default zero, maximum 1 GiB")
    parser.add_argument("--allowed-compressor-growth-bytes", type=int, default=None,
                        help="Use compressor physical peak growth instead of cumulative input; maximum 2 GiB")
    parser.add_argument("--expect-final-stopped", action="store_true")
    args = parser.parse_args()
    if not args.name or Path(args.name).name != args.name or not .5 <= args.interval <= 1 or not 1 <= args.duration <= 14400:
        parser.error("Use a plain name, interval0.5..1s and duration1..14400s")
    if not 0 <= args.allowed_global_swap_bytes <= DEFAULT_ALLOWED_SWAP_BYTES:
        parser.error("Swap allowance must be 0..104857600 bytes")
    if not 0 <= args.allowed_global_compression_bytes <= MAX_ALLOWED_COMPRESSION_BYTES:
        parser.error("Compression allowance must be 0..1073741824 bytes")
    if args.allowed_compressor_growth_bytes is not None:
        if not 0 <= args.allowed_compressor_growth_bytes <= MAX_ALLOWED_COMPRESSOR_GROWTH_BYTES:
            parser.error("Compressor growth allowance must be 0..2147483648 bytes")
        if args.allowed_global_compression_bytes != 0:
            parser.error("Select physical compressor growth or cumulative compression input, not both")
    base = args.output_dir.resolve()
    base.mkdir(parents=True, exist_ok=True, mode=0o700)
    trace, result, marker = (base / (args.name + suffix) for suffix in ("-samples.jsonl", "-summary.json", ".stop"))
    if result.exists() or marker.exists():
        parser.error("Output/stop marker already exists; choose a new name")
    if args.recheck_summary:
        summary = recheck_summary(args.recheck_summary,
            allowed_swap_bytes=args.allowed_global_swap_bytes,
            allowed_compression_bytes=args.allowed_global_compression_bytes,
            allowed_compressor_growth_bytes=args.allowed_compressor_growth_bytes,
            expect_final_stopped=args.expect_final_stopped)
        with result.open("x") as output:
            json.dump(summary, output, indent=2)
            output.write("\n")
        print(json.dumps({key: summary[key] for key in
            ("status", "errors", "swap_delta_bytes", "allowed_swap_bytes", "compression_delta_bytes", "compression_acceptance_metric", "allowed_compression_bytes", "allowed_compressor_growth_bytes", "compressor_physical_peak_growth_bytes", "final_service_stopped")}))
        return 0 if summary["status"] == "passed" else 1
    signal.signal(signal.SIGTERM, stop_signal)
    signal.signal(signal.SIGINT, stop_signal)
    provider, telemetry = Metadata(os.getuid()), MacTelemetry(reclaim_active_external=True)
    read_process = process_rusage_reader()
    rows, errors, previous_tasks, previous_activity = [], [], None, None
    started = time.monotonic()
    with trace.open("x") as stream:
        while True:
            began = time.monotonic()
            try:
                activity_before = provider.global_vm()
                tasks, task_errors = provider.tasks()
                activity = provider.global_vm()
                system = asdict(telemetry())
                system["observer_phys_footprint"] = system.pop("phys_footprint")
                pid = running()
                service = next((row for row in tasks if row["pid"] == pid), None) if pid else None
                if pid is not None and service is None:
                    # A new PID can appear just after enumeration; re-scan once.
                    tasks, task_errors = provider.tasks()
                    service = next((row for row in tasks if row["pid"] == pid), None)
                    if service is None and running() == pid:
                        raise RuntimeError("Owned live service task metadata unavailable")
                    if service is None:
                        pid = None
                if service is not None:
                    usage = read_process(pid)
                    if usage["proc_start_abstime"] != service["start_abstime"]:
                        raise RuntimeError("Service PID identity changed during rusage sample")
                    service["rusage_lifetime_max_phys_bytes"] = usage["lifetime_max_phys_footprint"]
                changes = task_deltas(tasks, previous_tasks) if previous_tasks is not None else []
                incremental = delta(activity, previous_activity or activity_before, GLOBAL_ACTIVITY)
                row = {"at": time.time(), "at_utc": datetime.now(timezone.utc).isoformat(),
                       "elapsed_s": began - started, "pid": pid, "service": service,
                       "system": system, "activity": activity, "swap_usage": swap_usage(provider),
                       "activity_delta_since_previous": incremental,
                       "same_window_task_activity": changes if any(incremental[key] for key in ("compressions", "swapins", "swapouts")) else [],
                       "task_access_error_count": len(task_errors)}
                if args.phase_file is not None:
                    try:
                        with args.phase_file.open() as phase_stream:
                            row["phase"] = phase_stream.read(128).strip()
                    except OSError:
                        row["phase"] = None
                rows.append(row)
                stream.write(json.dumps(row, separators=(",", ":")) + "\n")
                stream.flush()
                previous_tasks, previous_activity = tasks, activity
            except Exception as error:
                detail = type(error).__name__ + ": " + str(error)
                errors.append(detail)
                stream.write(json.dumps({"at": time.time(), "elapsed_s": time.monotonic() - started, "error": detail}) + "\n")
                stream.flush()
            if STOP or marker.exists() or time.monotonic() - started >= args.duration:
                break
            time.sleep(max(0, args.interval - (time.monotonic() - began)))
    summary = summarize(rows, errors, allowed_swap_bytes=args.allowed_global_swap_bytes,
                        allowed_compression_bytes=args.allowed_global_compression_bytes,
                        allowed_compressor_growth_bytes=args.allowed_compressor_growth_bytes,
                        expect_final_stopped=args.expect_final_stopped)
    summary.update(trace_path=str(trace), trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest(),
                   interval_seconds=args.interval, ended_by_signal=STOP, ended_by_marker=marker.exists())
    with result.open("x") as output:
        json.dump(summary, output, indent=2)
        output.write("\n")
    print(json.dumps({key: summary.get(key) for key in ("status", "errors", "sample_count", "activity_delta", "swap_delta_bytes", "allowed_swap_bytes", "compression_delta_bytes", "compression_acceptance_metric", "allowed_compression_bytes", "allowed_compressor_growth_bytes", "strict_zero_compression_gate_passed", "compressor_physical_delta_bytes", "compressor_physical_peak_growth_bytes", "strict_zero_paging_gate_passed", "final_service_stopped", "peak_sampled_process_phys_bytes")}))
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
