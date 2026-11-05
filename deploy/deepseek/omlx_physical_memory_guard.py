"""CPU-only physical-headroom admission and oMLX enforcer integration.

The startup estimate is an admission estimate, not proof of every workload's
peak. Runtime checks retain all existing process/Metal limits and reserve real
physical headroom without crediting anonymous, compressed, or swapped pages.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import functools
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import struct
import time

from v41_load_guard import GIB, MacTelemetry

AVAILABLE_BASIS = ("free + pageable file-backed cache (active and inactive); "
                   "reclaim estimate, not necessarily clean")


class PhysicalMemoryError(MemoryError):
    pass


@dataclass(frozen=True)
class MemoryPolicy:
    expected_resident_gib: float = 216
    load_transient_gib: float = 4
    # Legacy profiles retain an additional available-memory reserve until
    # explicitly migrated to the total-system-budget form.
    reserve_gib: float | None = 16
    activity_sample_seconds: float = 1
    system_budget_gib: float | None = None
    additional_free_reserve_gib: float | None = None

    @classmethod
    def from_profile(cls, profile):
        raw = profile.get("memory_policy")
        common = {"expected_resident_gib", "load_transient_gib", "activity_sample_seconds"}
        legacy = common | {"reserve_gib"}
        current = common | {"system_budget_gib", "additional_free_reserve_gib"}
        if not isinstance(raw, dict) or set(raw) not in (legacy, current, set(cls.__dataclass_fields__)):
            raise ValueError("Runtime profile lacks the complete physical memory policy")
        if set(raw) == set(cls.__dataclass_fields__):
            if raw["reserve_gib"] is not None:
                if raw["system_budget_gib"] is not None or raw["additional_free_reserve_gib"] is not None:
                    raise ValueError("Legacy and total-system memory policies cannot be combined")
                raw = {key: raw[key] for key in legacy}
            else:
                raw = {key: raw[key] for key in current}
        values = {}
        for name, value in raw.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("Invalid physical memory policy: " + name)
            values[name] = float(value)
        if set(raw) == current:
            values["reserve_gib"] = None
        policy = cls(**values)
        if (policy.expected_resident_gib <= 0 or policy.load_transient_gib < 0
                or not .1 <= policy.activity_sample_seconds <= 10
                or (policy.reserve_gib is not None and policy.reserve_gib < 16)
                or (policy.system_budget_gib is not None and policy.system_budget_gib < 16)
                or (policy.additional_free_reserve_gib is not None and policy.additional_free_reserve_gib < 0)):
            raise ValueError("Physical memory policy weakens required headroom or has invalid estimates")
        return policy

    @property
    def reserve_bytes(self):
        """Backward-compatible name for the additional-free reserve only."""
        return self.additional_free_reserve_bytes

    @property
    def additional_free_reserve_bytes(self):
        value = self.reserve_gib if self.reserve_gib is not None else self.additional_free_reserve_gib
        return int(value * GIB)

    @property
    def system_budget_bytes(self):
        # Legacy operation already had a 240 GiB cap on this 256 GiB host;
        # retain its additional reserve as well as that total-system budget.
        return int((16 if self.system_budget_gib is None else self.system_budget_gib) * GIB)

    def diagnostics(self):
        return {"system_budget_bytes": self.system_budget_bytes,
                "additional_free_reserve_bytes": self.additional_free_reserve_bytes,
                "reserve_bytes": self.reserve_bytes,
                "mode": "legacy_additional_free_reserve" if self.reserve_gib is not None else "total_system_budget"}

    @property
    def startup_required_bytes(self):
        return int((self.expected_resident_gib + self.load_transient_gib) * GIB) + self.reserve_bytes


def validate_snapshot(snapshot):
    values = (snapshot.phys_footprint, snapshot.available, snapshot.pressure, snapshot.total)
    if (any(type(value) is not int for value in values) or snapshot.total <= 0
            or not 0 < snapshot.phys_footprint <= snapshot.total
            or not 0 <= snapshot.available <= snapshot.total):
        raise ValueError("Invalid physical memory telemetry")
    return snapshot


def memory_components(snapshot):
    return {name: getattr(snapshot, name, None) for name in
            ("free_bytes", "speculative_bytes", "immediately_free_bytes",
             "inactive_external_bytes", "active_external_bytes", "credited_file_bytes",
             "vm_stats_words")}


def record_fast_activity(state, snapshot):
    """Record global activity; a single compressed page is not an abort rule."""
    current = {name: getattr(snapshot, name, None)
               for name in ("compressions", "decompressions", "swapins", "swapouts")}
    state["scope"] = "global counters since this guard's first observation; not service attribution"
    if any(type(value) is not int or value < 0 for value in current.values()):
        state["available"] = False
        state["reason"] = "fast activity counters unavailable in this snapshot"
        return
    if state.get("baseline") is None:
        state["baseline"] = current
    state["current"] = current
    state["delta"] = {name: current[name] - state["baseline"][name] for name in current}
    state["available"] = all(value >= 0 for value in state["delta"].values())
    if not state["available"]:
        state["reason"] = "global counters reset; original baseline retained"
    else:
        state.pop("reason", None)


def vm_activity(*, run=subprocess.check_output):
    """Read system counters; only the startup/check path invokes subprocesses."""
    output = run(["/usr/bin/vm_stat"], text=True)
    match = re.search(r"page size of (\d+) bytes", output)
    counts = {key.strip('"'): int(value) for key, value in
              re.findall(r"^([^:\n]+):\s+(\d+)\.", output, re.MULTILINE)}
    required = ("Compressions", "Decompressions", "Swapins", "Swapouts",
                "Pages occupied by compressor", "Pages stored in compressor")
    if not match or any(key not in counts for key in required):
        raise ValueError("Incomplete macOS compressor/swap telemetry")
    page_size = int(match.group(1))
    swap = run(["/usr/sbin/sysctl", "vm.swapusage"], text=True)
    used = re.search(r"used\s*=\s*([\d.]+)M", swap)
    if page_size <= 0 or not used:
        raise ValueError("Invalid macOS compressor/swap telemetry")
    return {"compressions": counts["Compressions"], "decompressions": counts["Decompressions"],
            "swapins": counts["Swapins"], "swapouts": counts["Swapouts"],
            "compressor_physical_bytes": counts["Pages occupied by compressor"] * page_size,
            "compressor_logical_bytes": counts["Pages stored in compressor"] * page_size,
            "swap_used_bytes": int(float(used.group(1)) * 1024**2)}


def check_startup(profile, *, telemetry=None, activity=vm_activity, sleep=time.sleep):
    """Fail closed before importing a model; old stored swap alone is allowed."""
    policy = MemoryPolicy.from_profile(profile)
    telemetry = telemetry or MacTelemetry(reclaim_active_external=True)
    try:
        first = validate_snapshot(telemetry())
        baseline = activity()
        sleep(policy.activity_sample_seconds)
        snapshot = validate_snapshot(telemetry())
        current = activity()
        deltas = {key: current[key] - baseline[key]
                  for key in ("compressions", "decompressions", "swapins", "swapouts")}
        if any(value < 0 for value in deltas.values()):
            raise ValueError("Memory counters reset during startup observation")
    except Exception as error:
        raise PhysicalMemoryError("Physical memory admission telemetry unavailable: "
                                  + type(error).__name__) from error
    reasons = []
    if first.pressure != 1 or snapshot.pressure != 1:
        reasons.append("system memory pressure is not normal")
    if min(first.available, snapshot.available) < policy.startup_required_bytes:
        reasons.append(f"requires {policy.startup_required_bytes / GIB:.2f} GiB available RAM estimate "
                       f"including {policy.reserve_bytes / GIB:g} GiB additional free reserve; "
                       f"observed {min(first.available, snapshot.available) / GIB:.2f} GiB")
    resident_load = int((policy.expected_resident_gib + policy.load_transient_gib) * GIB)
    if any(observed.phys_footprint + resident_load > observed.total - policy.system_budget_bytes
           for observed in (first, snapshot)):
        reasons.append("resident/load estimate would exceed the total-system memory budget")
    if deltas["swapins"] or deltas["swapouts"] or deltas["compressions"]:
        reasons.append("new compression or swap activity observed; retry after memory settles")
    result = {"admissible": not reasons, "policy": asdict(policy), **policy.diagnostics(),
              "physical_available_basis": AVAILABLE_BASIS,
              "startup_estimate_only": True,
              "total_bytes": snapshot.total, "available_bytes": snapshot.available,
              "required_available_bytes": policy.startup_required_bytes,
              "pressure": snapshot.pressure, "baseline": baseline, "current": current,
              "activity_delta": deltas, "memory_components": memory_components(snapshot),
              "reasons": reasons}
    if reasons:
        raise PhysicalMemoryError("oMLX startup refused: " + "; ".join(reasons))
    return result


def wait_for_startup_memory(profile, *, timeout_seconds=30, quiet_windows=2,
                            telemetry=None, activity=vm_activity, sleep=time.sleep,
                            monotonic=time.monotonic):
    """Wait only before model imports; never retry a partially loaded model.

    Every candidate window keeps check_startup's full memory/pressure/activity
    requirements. A failed window resets the consecutive-quiet count, while
    old insufficient samples do not remain the minimum of future windows.
    """
    policy = MemoryPolicy.from_profile(profile)
    if (isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds) or timeout_seconds <= 0
            or type(quiet_windows) is not int or quiet_windows < 1):
        raise ValueError("Invalid bounded memory settling window")
    telemetry = telemetry or MacTelemetry(reclaim_active_external=True)
    started = monotonic()
    deadline = started + timeout_seconds
    quiet = attempts = 0
    latest_reason = "no complete memory observation window was available"
    transient_refusal = None
    while monotonic() + policy.activity_sample_seconds <= deadline:
        attempts += 1
        attempt_started = monotonic()
        try:
            report = check_startup(profile, telemetry=telemetry, activity=activity, sleep=sleep)
        except PhysicalMemoryError as error:
            quiet = 0
            transient_refusal = latest_reason = str(error)
        else:
            quiet += 1
            latest_reason = f"only {quiet}/{quiet_windows} consecutive quiet observation windows"
            if quiet >= quiet_windows:
                report["settling"] = {"attempts": attempts, "quiet_windows": quiet,
                    "elapsed_seconds": monotonic() - started,
                    "timeout_seconds": timeout_seconds, "latest_transient_refusal": transient_refusal}
                return report
        # Telemetry can fail before check_startup reaches its sample sleep;
        # retain a bounded cadence instead of spinning on an early failure.
        missing_interval = policy.activity_sample_seconds - (monotonic() - attempt_started)
        if missing_interval > 0:
            sleep(min(missing_interval, max(0, deadline - monotonic())))
    remaining = max(0, deadline - monotonic())
    if remaining:
        sleep(remaining)
    raise PhysicalMemoryError(f"oMLX memory did not settle within {timeout_seconds:g} seconds: " + latest_reason)


def physical_ceiling(snapshot, policy):
    snapshot = validate_snapshot(snapshot)
    # A nonzero cap preserves the enforcer's "enabled" semantics even when
    # telemetry or pressure requires its established emergency/reclaim path.
    cap = min(snapshot.total - policy.system_budget_bytes,
              snapshot.phys_footprint + snapshot.available - policy.additional_free_reserve_bytes)
    if snapshot.pressure != 1:
        # Preserve the emergency brake even when the additional reserve is 0.
        brake = max(policy.system_budget_bytes, policy.additional_free_reserve_bytes, 1)
        cap = min(cap, snapshot.phys_footprint - brake)
    return max(1, cap)


def install_runtime_guard(enforcer_class, profile, *, telemetry=None):
    """Cap admission AND active-request braking; import no oMLX/MLX here."""
    policy = MemoryPolicy.from_profile(profile)
    telemetry = telemetry or MacTelemetry(reclaim_active_external=True)
    if getattr(enforcer_class, "_physical_headroom_guard_installed", False):
        raise RuntimeError("Physical memory guard is already installed")
    original_breakdown = enforcer_class._get_ceiling_breakdown
    original_abort = enforcer_class._get_abort_limit_bytes
    original_admission = getattr(enforcer_class, "get_admission_ceiling", None)
    state = {"enabled": True, **policy.diagnostics(),
             "physical_available_basis": AVAILABLE_BASIS,
             "policy": asdict(policy), "last_check": None, "activity": {}}

    def cap():
        try:
            snapshot = telemetry()
            ceiling = physical_ceiling(snapshot, policy)
            record_fast_activity(state["activity"], snapshot)
            state["last_check"] = {"at": time.time(), "available_bytes": snapshot.available,
                                   "process_phys_bytes": snapshot.phys_footprint,
                                   "pressure": snapshot.pressure, "physical_cap_bytes": ceiling,
                                   "system_budget_cap_bytes": max(1, snapshot.total - policy.system_budget_bytes),
                                   "available_headroom_cap_bytes": max(1, snapshot.phys_footprint + snapshot.available
                                                                        - policy.additional_free_reserve_bytes),
                                   "memory_components": memory_components(snapshot)}
            return ceiling
        except Exception as error:
            state["last_check"] = {"at": time.time(), "physical_cap_bytes": 1,
                                   "telemetry_error": type(error).__name__}
            return 1

    @functools.wraps(original_breakdown)
    def breakdown(self):
        values = dict(original_breakdown(self))
        ceiling = cap()
        values["dynamic"] = min(values["dynamic"], ceiling) if values["dynamic"] > 0 else ceiling
        values["hard_limit"] = min(values["hard_limit"], ceiling) if values["hard_limit"] > 0 else ceiling
        return values

    @functools.wraps(original_abort)
    def abort_limit(self):
        original = original_abort(self)
        ceiling = cap()
        return min(original, ceiling) if original > 0 else ceiling

    if original_admission is not None:
        @functools.wraps(original_admission)
        def admission_limit(self):
            original = original_admission(self)
            ceiling = cap()
            return min(original, ceiling) if original > 0 else ceiling

        enforcer_class.get_admission_ceiling = admission_limit

    enforcer_class._get_ceiling_breakdown = breakdown
    enforcer_class._get_abort_limit_bytes = abort_limit
    enforcer_class._physical_headroom_guard_installed = True
    return state


def _file_identity(stat):
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def native_compact_residency(plan):
    """Count the verified compact payload from bounded headers, including MTP.

    A fully validated NativeMetadataPlan is required. Its original estimator
    stays untouched: only the final admission wrapper replaces its blanket 5%
    allowance with the explicit loading allowance under synchronous checks.
    """
    checkpoint = Path(plan.checkpoint).resolve(strict=True)
    proof_path = Path(plan.proof_path).resolve(strict=True)
    metadata_paths = (checkpoint / "config.json", checkpoint / "model.safetensors.index.json",
                      proof_path, proof_path.parent / "DeepSeek-V4.1-LSQ3-G128-exact-metadata.json")
    metadata_identities = {path: _file_identity(path.stat()) for path in metadata_paths}
    with proof_path.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != plan.proof_sha256:
            raise ValueError("Native metadata proof changed before admission")
    config_raw = (checkpoint / "config.json").read_bytes()
    index_raw = (checkpoint / "model.safetensors.index.json").read_bytes()
    sidecar = json.loads(metadata_paths[-1].read_text())
    if (hashlib.sha256(config_raw).hexdigest() != sidecar["source_config_sha256"]
            or hashlib.sha256(index_raw).hexdigest() != sidecar["source_index_sha256"]):
        raise ValueError("Compact admission config/index differs from validated metadata")
    config = json.loads(config_raw)
    spec = config.get("omlx_deepseek_v41") or {}
    if spec.get("version") != 1 or spec.get("preserve_mtp") is not True:
        raise ValueError("Compact admission requires all preserved MTP tensors")
    tables = spec.get("engram_tables")
    if not isinstance(tables, dict) or not tables:
        raise ValueError("Compact admission requires explicit Engram tables")
    engram_keys = {table[field] for table in tables.values()
                   for field in ("weight_key", "scale_key", "bias_key") if table.get(field)}
    mapping = json.loads(index_raw)["weight_map"]
    headers = {}
    base = mtp = 0
    widths = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "U8": 1, "I8": 1,
              "U16": 2, "I16": 2, "U32": 4, "I32": 4, "U64": 8, "I64": 8, "BOOL": 1}
    for key, name in mapping.items():
        if key in engram_keys:
            continue
        path = (checkpoint / name).resolve(strict=True)
        if not path.is_relative_to(checkpoint):
            raise ValueError("Compact admission shard escapes checkpoint")
        if path not in headers:
            with path.open("rb") as stream:
                before = os.fstat(stream.fileno())
                if plan.source_identities.get(path) != _file_identity(before):
                    raise ValueError("Compact admission source identity changed")
                raw = stream.read(8)
                if len(raw) != 8:
                    raise ValueError("Truncated compact admission header")
                length = struct.unpack("<Q", raw)[0]
                if not 2 <= length <= min(16 * 1024**2, before.st_size - 8):
                    raise ValueError("Invalid compact admission header extent")
                header = stream.read(length)
                if len(header) != length or _file_identity(before) != _file_identity(os.fstat(stream.fileno())):
                    raise ValueError("Compact admission header changed while reading")
                headers[path] = (json.loads(header), 8 + length, before.st_size)
        header, offset, file_size = headers[path]
        item = header[key]
        begin, end = item["data_offsets"]
        shape = item["shape"]
        if (type(begin) is not int or type(end) is not int or begin < 0 or end < begin
                or offset + end > file_size or not isinstance(shape, list)
                or any(type(value) is not int or value < 0 for value in shape)
                or item["dtype"] not in widths
                or end - begin != math.prod(shape) * widths[item["dtype"]]):
            raise ValueError("Invalid compact admission tensor extent")
        base += end - begin
        if key.startswith("language_model.mtp.") or key.startswith("mtp."):
            mtp += end - begin
    original_metadata = sum(record.original_bytes for record in plan.records.values())
    encoded_metadata = sum(record.encoded_bytes for record in plan.records.values())
    saved = original_metadata - encoded_metadata
    if not 0 < saved == plan.saved_bytes < base or mtp <= 0:
        raise ValueError("Invalid compact admission metadata or MTP accounting")
    compact = base - saved
    if type(plan.prefill_transient_bytes) is not int or plan.prefill_transient_bytes < 0:
        raise ValueError("Invalid compact admission transient accounting")
    # Match the unchanged legacy estimator exactly, including both rounding
    # sites; subtracting its saving a second time would undercount residency.
    legacy = int(base * 1.05) + 32 * 1024**2 - int(saved * 1.05)
    for path in headers:
        if plan.source_identities.get(path) != _file_identity(path.stat()):
            raise ValueError("Compact admission source changed after inspection")
    with proof_path.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != plan.proof_sha256:
            raise ValueError("Native metadata proof changed during admission")
    if any(_file_identity(path.stat()) != identity for path, identity in metadata_identities.items()):
        raise ValueError("Compact admission metadata changed during inspection")
    return {"original_base_bytes": base, "compact_tensor_bytes": compact,
            "saved_metadata_bytes": saved, "legacy_mmap_estimate_bytes": legacy,
            "prefill_transient_bytes": plan.prefill_transient_bytes, "mtp_tensor_bytes": mtp}


def install_native_load_guard(plan_class, engine_pool_class, profile, *, telemetry=None):
    """Guard each native allocation and narrowly correct final admission.

    This helper imports no inference framework. The launcher supplies the
    already imported classes inside the independent service process.
    """
    policy = MemoryPolicy.from_profile(profile)
    telemetry = telemetry or MacTelemetry(reclaim_active_external=True)
    checkpoint = Path(profile["model_path"]).resolve(strict=True)
    proof_path = Path(profile["metadata_proof"]).resolve(strict=True)
    effective = profile["effective_settings"]
    limit_values = (profile["process_limit_gib"] * GIB,
                    effective["memory_guard_gb"] * GIB,
                    effective["wired_limit_mib"] * 1024**2)
    if any(isinstance(value, bool) or not math.isfinite(value) or value <= 0 for value in limit_values):
        raise ValueError("Native load guard requires existing process and Metal caps")
    existing_cap = int(min(limit_values))
    if getattr(plan_class, "_physical_allocation_guard_installed", False):
        raise RuntimeError("Native allocation guard is already installed")
    if getattr(engine_pool_class, "_compact_admission_guard_installed", False):
        raise RuntimeError("Compact admission guard is already installed")
    original_init = plan_class.__init__
    original_estimate = engine_pool_class._entry_runtime_resident_size
    state = {"enabled": True, **policy.diagnostics(),
             "existing_cap_bytes": existing_cap, "allocation_checks": 0,
             "last_allocation": None, "latest_accounting": None,
             "physical_available_basis": AVAILABLE_BASIS}

    @functools.wraps(original_init)
    def initialize(plan, checkpoint_arg, proof_arg, *, allocation_check=None):
        original_init(plan, checkpoint_arg, proof_arg, allocation_check=allocation_check)
        if (Path(checkpoint_arg).resolve() != checkpoint or Path(proof_arg).resolve() != proof_path):
            return
        previous = plan.allocation_check
        latched = None
        load_activity = {}

        def checked(requested, *, stage="native load"):
            nonlocal latched
            if latched is not None:
                raise PhysicalMemoryError(latched)
            try:
                if type(requested) is not int or requested < 0:
                    raise ValueError("Invalid native allocation estimate")
                original_requested = requested
                requested = (requested + 16383) // 16384 * 16384
                snapshot = validate_snapshot(telemetry())
                cap = min(existing_cap, physical_ceiling(snapshot, policy))
                record_fast_activity(load_activity, snapshot)
                state["activity"] = load_activity
                state["allocation_checks"] += 1
                state["last_allocation"] = {"stage": str(stage)[:160], "requested_bytes": requested,
                    "process_phys_bytes": snapshot.phys_footprint, "available_bytes": snapshot.available,
                    "pressure": snapshot.pressure, "effective_cap_bytes": cap,
                    "memory_components": memory_components(snapshot)}
                if snapshot.pressure != 1:
                    raise PhysicalMemoryError("System memory pressure is not normal")
                if snapshot.phys_footprint + requested > cap:
                    raise PhysicalMemoryError("Native allocation would exceed physical headroom or an existing cap")
                if previous is not None:
                    previous(original_requested, stage=stage)
            except Exception as error:
                latched = ("Native load refused and latched: " + type(error).__name__
                           + ": " + str(error))
                state["last_refusal"] = latched
                raise PhysicalMemoryError(latched) from error

        plan.allocation_check = checked

    @functools.wraps(original_estimate)
    def estimate(pool, entry, runtime_settings, *, base_size=None, include_ane_reservation=True):
        original = original_estimate(pool, entry, runtime_settings, base_size=base_size,
                                     include_ane_reservation=include_ane_reservation)
        active_proof = os.environ.get("DS41_NATIVE_METADATA")
        if (getattr(entry, "model_id", None) != profile["model_id"]
                or Path(entry.model_path).resolve() != checkpoint
                or getattr(entry, "config_model_type", "") != "deepseek_v41"
                or getattr(entry, "engine_type", None) != "vlm"
                or not getattr(runtime_settings, "mtp_enabled", False)
                or not getattr(runtime_settings, "deepseek_v41_engram_ssd_offload", False)
                or getattr(runtime_settings, "moe_expert_offload_enabled", False)
                or not active_proof or Path(active_proof).resolve() != proof_path
                or pool._distributed_deployment_for_entry(entry) is not None
                or pool.external_engine_id(entry.model_id, runtime_settings)):
            return original
        # Fully validate original/encoded source identities before replacing
        # only the native compact component of the final admission estimate.
        plan = plan_class(checkpoint, proof_path)
        accounting = native_compact_residency(plan)
        allowance = int(policy.load_transient_gib * GIB)
        if allowance < accounting["prefill_transient_bytes"]:
            raise PhysicalMemoryError("Explicit native load allowance is smaller than a dense metadata bank")
        legacy = accounting["legacy_mmap_estimate_bytes"]
        if type(original) is not int or original < legacy:
            return original
        corrected = (original - legacy + accounting["compact_tensor_bytes"]
                     + allowance + 32 * 1024**2)
        state["latest_accounting"] = {**accounting, "original_estimate_bytes": original,
            "corrected_estimate_bytes": corrected, "explicit_load_allowance_bytes": allowance,
            "other_runtime_reservations_bytes": original - legacy}
        # A larger explicit allowance may legitimately require more memory.
        # Never hide it by returning a smaller estimate than its actual charge.
        return corrected

    plan_class.__init__ = initialize
    engine_pool_class._entry_runtime_resident_size = estimate
    plan_class._physical_allocation_guard_installed = True
    engine_pool_class._compact_admission_guard_installed = True
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(check_startup(json.loads(args.profile.read_text())), indent=2))
    except (OSError, ValueError, PhysicalMemoryError) as error:
        parser.exit(1, str(error) + "\n")


if __name__ == "__main__":
    main()
