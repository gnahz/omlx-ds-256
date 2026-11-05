"""Synchronous, fail-closed physical-memory checks for compact V4.1 loads.

The asynchronous engine-pool enforcer cannot interrupt a loader holding its
lock. This guard runs on the loading thread before allocations and between
bounded file reads. It changes no Metal limits or normal .95/.97 thresholds.
The module itself imports neither MLX nor oMLX; providers are injectable.
"""

from dataclasses import dataclass
import ctypes
import logging
import math
import os
import sys
from threading import Lock

GIB = 1024**3
RESERVE_FLAG = "LOCAL_MOE_V41_LOAD_RESERVE_GIB"
CEILING_FLAG = "LOCAL_MOE_V41_LOAD_MAX_GIB"
MIN_RESERVE = 16 * GIB
# Current installed SDK: sizeof(vm_statistics64_data_t)/sizeof(integer_t).
# Oversized requested counts105/128/256 yield a legacy40-word reply on this
# host, even though104 yields104. Spare capacity is only for bounded retries.
VM_STATS_REQUEST_WORDS = 104
VM_STATS_BUFFER_WORDS = 256
logger = logging.getLogger(__name__)


class LoadGuardError(MemoryError):
    """Recoverable load refusal; a new service process may retry explicitly."""


@dataclass(frozen=True)
class Snapshot:
    phys_footprint: int
    available: int
    pressure: int
    total: int
    free_bytes: int | None = None
    inactive_external_bytes: int | None = None
    active_external_bytes: int | None = None
    credited_file_bytes: int | None = None
    compressions: int | None = None
    decompressions: int | None = None
    swapins: int | None = None
    swapouts: int | None = None
    vm_stats_words: int | None = None
    speculative_bytes: int | None = None
    immediately_free_bytes: int | None = None


@dataclass(frozen=True)
class Limits:
    process: int
    metal: int


def preflight_estimate(snapshot, *, additional_resident_bytes, transient_bytes=0,
                       reserve_bytes=MIN_RESERVE, limits=None):
    """Read-only estimate; never reads providers, changes limits, or latches.

    Pass raw additional resident storage plus an explicit load-transient
    budget. If the caller already passes an admission estimate containing a
    reserve, leave transient_bytes=0 to avoid counting that reserve twice.
    Existing process/Metal limits are optional; physical_fit alone does not
    assert the process can be admitted when they were not supplied.
    """
    for name, value in (("additional resident bytes", additional_resident_bytes),
                        ("transient bytes", transient_bytes), ("reserve bytes", reserve_bytes)):
        if type(value) is not int or value < 0:
            raise ValueError(f"Invalid {name}")
    if reserve_bytes < MIN_RESERVE:
        raise ValueError("Compact load reserve must be at least 16 GiB")
    requested = additional_resident_bytes + transient_bytes
    physical_shortfall = max(0, requested + reserve_bytes - snapshot.available)
    existing_ceiling = None if limits is None else min(limits.process, limits.metal)
    ceiling_shortfall = None if existing_ceiling is None else max(
        0, snapshot.phys_footprint + requested - existing_ceiling)
    physical_fit = snapshot.pressure == 1 and physical_shortfall == 0
    return {
        "units": "bytes; GiB=2**30", "pressure": snapshot.pressure,
        "physical_available_basis": "free + inactive file-backed lower bound",
        "available_bytes": snapshot.available, "process_phys_bytes": snapshot.phys_footprint,
        "additional_resident_bytes": additional_resident_bytes, "transient_bytes": transient_bytes,
        "reserve_bytes": reserve_bytes, "required_available_bytes": requested + reserve_bytes,
        "predicted_available_bytes": snapshot.available - requested,
        "physical_shortfall_bytes": physical_shortfall, "physical_fit": physical_fit,
        "effective_existing_ceiling_bytes": existing_ceiling,
        "ceiling_shortfall_bytes": ceiling_shortfall,
        "admissible": None if limits is None else physical_fit and ceiling_shortfall == 0,
    }


class _RusagePrefix(ctypes.Structure):
    # Stable rusage_info_v4 prefix (sys/resource.h); the tail is oversized so
    # proc_pid_rusage can fill its complete v4 structure. phys is at byte72.
    _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [
        (name, ctypes.c_uint64) for name in (
            "user", "system", "idle_wakeups", "interrupt_wakeups", "pageins",
            "wired", "resident", "phys_footprint")
    ] + [("tail", ctypes.c_uint64 * 64)]


def _available_breakdown(stats, count, page_size, *, reclaim_active_external=False):
    """Free plus file-backed queue credit, never anonymous pages.

    Current macOS SDK vm_statistics64_data_t: internal_page_count is U32 at
    byte140; inactive_external_count is U64 at byte336 (rev5). Gate on the
    returned count, not buffer capacity. Optional active credit additionally
    uses active_external_count at byte328. Older replies retain the historical
    inactive - all_internal lower bound; absent that field, credit free only.
    """
    free = ctypes.c_uint32(stats[0]).value
    active = ctypes.c_uint32(stats[1]).value
    inactive = ctypes.c_uint32(stats[2]).value
    # The SDK explicitly includes speculative file pages in free_count.
    # Preserve the availability formula, while exposing actual free pages
    # separately so queue aging cannot look like a new large allocation.
    speculative = min(free, ctypes.c_uint32(stats[23]).value) if count * 4 >= 96 else None
    external_active = 0
    if count * 4 >= 344:
        external_inactive = ctypes.c_uint64.from_buffer(stats, 336).value
        reclaimable = min(inactive, external_inactive)
        if reclaim_active_external:
            external_active = min(active, ctypes.c_uint64.from_buffer(stats, 328).value)
    elif count * 4 >= 144:
        internal = ctypes.c_uint32(stats[35]).value
        reclaimable = max(0, inactive - internal)
    else:
        reclaimable = 0
    return {"free_bytes": free * page_size,
            "speculative_bytes": None if speculative is None else speculative * page_size,
            "immediately_free_bytes": None if speculative is None else (free - speculative) * page_size,
            "inactive_external_bytes": reclaimable * page_size,
            "active_external_bytes": external_active * page_size,
            "credited_file_bytes": (reclaimable + external_active) * page_size}


def _available_bytes(stats, count, page_size, *, reclaim_active_external=False):
    """Historical inactive-only estimate unless active file cache is opted in.

    Active credit requires returned rev5 fields; raw active/anonymous pages
    never count. File-backed pageable pages may require writeback and are not
    necessarily clean. Speculative and purgeable counters are not added again.
    """
    values = _available_breakdown(stats, count, page_size,
                                  reclaim_active_external=reclaim_active_external)
    return values["free_bytes"] + values["credited_file_bytes"]


def _activity_counts(stats, count):
    """Fast cumulative counters from the returned ABI extent, without shells."""
    result = {}
    for name, offset in (("decompressions", 96), ("compressions", 104),
                         ("swapins", 112), ("swapouts", 120)):
        result[name] = (ctypes.c_uint64.from_buffer(stats, offset).value
                        if count * 4 >= offset + 8 else None)
    return result


def read_host_vm_stats(libc, host):
    """Request the SDK-sized ABI; always decode only the returned extent.

    The kernel may cache replies. A single MIG_ARRAY_TOO_LARGE retry preserves
    the existing4..256-word bound; no counter formula or limit changes here.
    """
    stats = (ctypes.c_int * VM_STATS_BUFFER_WORDS)()
    count = ctypes.c_uint(VM_STATS_REQUEST_WORDS)
    rc = libc.host_statistics64(host, 4, stats, ctypes.byref(count))
    retried = False
    if rc == -307 and 4 <= count.value <= VM_STATS_BUFFER_WORDS:
        retried = True
        rc = libc.host_statistics64(host, 4, stats, ctypes.byref(count))
    if rc or not 4 <= count.value <= VM_STATS_BUFFER_WORDS:
        raise RuntimeError("Cannot read live host memory counters")
    return stats, count.value, retried


class MacTelemetry:
    """Direct libproc/sysctl/host reads; host_statistics64 may be kernel-cached.

    The returned ABI count controls which fields can be used. No shell or GPU
    work is performed by this provider.
    """

    def __init__(self, *, reclaim_active_external=False):
        if sys.platform != "darwin":
            raise RuntimeError("Compact load guard requires macOS telemetry")
        if type(reclaim_active_external) is not bool:
            raise TypeError("Active file cache reclaim policy must be explicit")
        self.reclaim_active_external = reclaim_active_external
        self.libproc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
        self.libc = ctypes.CDLL("/usr/lib/libc.dylib", use_errno=True)
        self.libproc.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        self.libproc.proc_pid_rusage.restype = ctypes.c_int
        self.libc.sysctlbyname.argtypes = [ctypes.c_char_p, ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_size_t), ctypes.c_void_p, ctypes.c_size_t]
        self.libc.sysctlbyname.restype = ctypes.c_int
        self.libc.mach_host_self.restype = ctypes.c_uint
        self.libc.host_page_size.argtypes = [ctypes.c_uint, ctypes.POINTER(ctypes.c_size_t)]
        self.libc.host_statistics64.argtypes = [ctypes.c_uint, ctypes.c_int, ctypes.c_void_p,
                                               ctypes.POINTER(ctypes.c_uint)]
        self.host = self.libc.mach_host_self()
        page = ctypes.c_size_t()
        if self.libc.host_page_size(self.host, ctypes.byref(page)) or page.value <= 0:
            raise RuntimeError("Cannot read macOS page size")
        self.page_size = page.value

    def sysctl(self, name):
        value = ctypes.c_uint64()
        size = ctypes.c_size_t(ctypes.sizeof(value))
        if self.libc.sysctlbyname(name.encode(), ctypes.byref(value), ctypes.byref(size), None, 0):
            raise RuntimeError("Cannot read required memory sysctl")
        if size.value == 4:
            return ctypes.c_int32(value.value).value
        if size.value == 8:
            return value.value
        raise RuntimeError("Unexpected memory sysctl width")

    def __call__(self):
        pressure = self.sysctl("kern.memorystatus_vm_pressure_level")
        usage = _RusagePrefix()
        if self.libproc.proc_pid_rusage(os.getpid(), 4, ctypes.byref(usage)):
            raise RuntimeError("Cannot read process phys_footprint")
        stats, count, _ = read_host_vm_stats(self.libc, self.host)
        components = _available_breakdown(stats, count, self.page_size,
            reclaim_active_external=self.reclaim_active_external)
        available = components["free_bytes"] + components["credited_file_bytes"]
        later_pressure = self.sysctl("kern.memorystatus_vm_pressure_level")
        if later_pressure != 1:
            pressure = later_pressure
        return Snapshot(int(usage.phys_footprint), available, pressure,
                        self.sysctl("hw.memsize"), **components,
                        **_activity_counts(stats, count), vm_stats_words=count)


class RuntimeLimits:
    """Read current oMLX settings and the effective Metal cap without changing either."""

    def __init__(self, telemetry):
        self.telemetry = telemetry
        self.recommended = None

    def __call__(self):
        from omlx.settings import get_settings
        from omlx.process_memory_enforcer import tier_reserve_bytes
        total = self.telemetry.sysctl("hw.memsize")
        try:
            memory = get_settings().memory
            ceiling = total - tier_reserve_bytes(memory.memory_guard_tier, total)
            if memory.memory_guard_tier == "custom":
                custom = int(memory.memory_guard_custom_ceiling_gb * GIB)
                if custom <= 0:
                    raise ValueError("Invalid current custom memory ceiling")
                ceiling = min(ceiling, custom)
        except RuntimeError:
            # Standalone tools have no server settings singleton. This is the
            # run_server.sh default, or its existing lower requested ceiling.
            requested = _gib(os.environ.get("LOCAL_MOE_MEMORY_GUARD_GB", "220"), "server ceiling")
            ceiling = min(total - 2 * GIB, requested, 228 * GIB)
        wired_mib = self.telemetry.sysctl("iogpu.wired_limit_mb")
        if wired_mib < 0:
            raise ValueError("Invalid effective Metal cap")
        if wired_mib:
            metal = wired_mib * 1024**2
        else:
            if self.recommended is None:
                import mlx.core as mx
                self.recommended = int(mx.device_info().get("max_recommended_working_set_size", 0))
            metal = self.recommended
        return Limits(int(ceiling), int(metal))


def _gib(value, label):
    try:
        number = float(value)
    except (ValueError, TypeError):
        raise ValueError(f"Invalid {label}") from None
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"Invalid {label}")
    return int(number * GIB)


class LoadGuard:
    """A rejection latches for this instance; checks never allocate Metal data."""

    def __init__(self, snapshot_provider, limits_provider, *, environment=None):
        env = os.environ if environment is None else environment
        self.reserve = _gib(env.get(RESERVE_FLAG, "16"), "load reserve")
        if self.reserve < MIN_RESERVE:
            raise ValueError("Compact load reserve must be at least 16 GiB")
        value = env.get(CEILING_FLAG)
        self.requested_ceiling = None if value is None else _gib(value, "load ceiling")
        self.snapshot_provider = snapshot_provider
        self.limits_provider = limits_provider
        self.latched = None
        self.last_snapshot = None
        self.last_limits = None

    def _reject(self, reason, allocation, stage):
        snap, limits = self.last_snapshot, self.last_limits
        message = (f"Compact V4.1 load refused: {reason}; stage={stage}; allocation={allocation}; "
                   f"phys={getattr(snap, 'phys_footprint', None)}; "
                   f"available={getattr(snap, 'available', None)}; "
                   f"pressure={getattr(snap, 'pressure', None)}; reserve={self.reserve}; "
                   f"process_ceiling={getattr(limits, 'process', None)}; "
                   f"metal_ceiling={getattr(limits, 'metal', None)}. "
                   "Compact loading is latched off in this process; restart explicitly after resolving pressure.")
        self.latched = message
        logger.error(message)
        raise LoadGuardError(message)

    def check(self, allocation=0, *, stage="load"):
        """Before allocation charge its full size; after allocation charge zero.

        Availability credits free pages and a lower bound on inactive external
        file pages; anonymous inactive and active file pages are not credited.
        No additional buffer reuse or other-process compression is assumed.
        Comparing the entire process footprint to the Metal ceiling
        is intentionally stricter than the driver's GPU-only limit.
        """
        if self.latched is not None:
            raise LoadGuardError(self.latched)
        if type(allocation) is not int or allocation < 0:
            raise ValueError("Allocation estimate must be a nonnegative integer")
        # The supported Apple Silicon host has 16 KiB physical pages. Rounding
        # each new allocation upward avoids undercharging small tensor tails.
        allocation = (allocation + 16383) // 16384 * 16384
        self.last_snapshot = self.last_limits = None
        # Stages are developer-controlled labels, never configuration/requests.
        stage = str(stage).replace("\n", " ").replace("\r", " ")[:160]
        try:
            snap = self.snapshot_provider()
            self.last_snapshot = snap
            if (any(type(n) is not int for n in (snap.phys_footprint, snap.available, snap.pressure, snap.total))
                    or snap.phys_footprint <= 0 or snap.total <= 0
                    or not 0 <= snap.available <= snap.total or snap.phys_footprint > snap.total):
                raise ValueError("Invalid physical memory telemetry")
        except Exception as exc:
            self._reject(f"memory telemetry unavailable ({type(exc).__name__})", allocation, stage)
        if snap.pressure != 1:
            self._reject("system memory pressure is not normal", allocation, stage)
        try:
            limits = self.limits_provider()
            self.last_limits = limits
            if any(type(n) is not int or n <= 0 for n in (limits.process, limits.metal)):
                raise ValueError("Invalid existing memory ceiling")
        except Exception as exc:
            self._reject(f"memory ceiling unavailable ({type(exc).__name__})", allocation, stage)
        ceiling = min(limits.process, limits.metal)
        if self.requested_ceiling is not None:
            if self.requested_ceiling > ceiling:
                self._reject("requested load ceiling would raise an existing limit", allocation, stage)
            ceiling = self.requested_ceiling
        if snap.phys_footprint + allocation > ceiling:
            self._reject("predicted process allocation exceeds an existing ceiling", allocation, stage)
        if snap.available - allocation < self.reserve:
            self._reject("predicted system available memory would fall below reserve", allocation, stage)
        return snap


_default_guard = None
_default_lock = Lock()


def compact_load_guard():
    """One process-wide instance prevents automatic fallback/reload loops."""
    global _default_guard
    if _default_guard is None:
        with _default_lock:
            if _default_guard is None:
                try:
                    if os.environ.get('LOCAL_MOE_V41_NATIVE_TELEMETRY') == '1':
                        from v41_native_telemetry import NativeMacTelemetry
                        telemetry = NativeMacTelemetry()
                    else:
                        telemetry = MacTelemetry()
                    _default_guard = LoadGuard(telemetry, RuntimeLimits(telemetry))
                except Exception as exc:
                    # Fail closed even before a functioning telemetry provider
                    # exists, and retain the same refusal for later attempts.
                    kind = type(exc).__name__
                    def unavailable():
                        raise RuntimeError(kind)
                    _default_guard = LoadGuard(unavailable, unavailable, environment={})
                    _default_guard._reject(f"load guard initialization failed ({kind})", 0, "initialize")
    return _default_guard
