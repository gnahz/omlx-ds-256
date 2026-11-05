"""Bounded, read-only process footprint observation from macOS libproc.

Uses rusage_info_v4 from the installed macOS SDK sys/resource.h. This is
physical-footprint evidence, not an MLX allocation or GPU-memory estimator.
It does not change limits, terminate processes, or inspect process arguments.
"""

import argparse
import ctypes
import json
from pathlib import Path
import time

FIELDS = """user_time system_time pkg_idle_wkups interrupt_wkups pageins wired_size
resident_size phys_footprint proc_start_abstime proc_exit_abstime child_user_time
child_system_time child_pkg_idle_wkups child_interrupt_wkups child_pageins
child_elapsed_abstime diskio_bytesread diskio_byteswritten cpu_time_qos_default
cpu_time_qos_maintenance cpu_time_qos_background cpu_time_qos_utility cpu_time_qos_legacy
cpu_time_qos_user_initiated cpu_time_qos_user_interactive billed_system_time
serviced_system_time logical_writes lifetime_max_phys_footprint instructions cycles
billed_energy serviced_energy interval_max_phys_footprint runnable_time""".split()


class RusageV4(ctypes.Structure):
    _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [(name, ctypes.c_uint64) for name in FIELDS]


def reader():
    library = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
    function = library.proc_pid_rusage
    function.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_void_p)
    function.restype = ctypes.c_int
    def read(pid):
        data = RusageV4()
        if function(pid, 4, ctypes.byref(data)):
            raise OSError(ctypes.get_errno(), "proc_pid_rusage failed")
        return {key: getattr(data, key) for key in (
            "phys_footprint", "lifetime_max_phys_footprint", "resident_size",
            "wired_size", "diskio_bytesread", "diskio_byteswritten", "pageins",
            "proc_start_abstime")}
    return read


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", required=True, type=int)
    parser.add_argument("--seconds", type=int, default=180)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--system", action="store_true",
                        help="Also record conservative macOS available memory and pressure")
    args = parser.parse_args()
    if args.pid <= 0 or not 1 <= args.seconds <= 1800:
        parser.error("Positive PID and duration1..1800 seconds required")
    read = reader()
    first = read(args.pid)
    if args.system:
        from v41_load_guard import MacTelemetry
        system = MacTelemetry()
    else:
        system = None
    started = time.monotonic()
    with args.output.open("x") as file:
        file.write(json.dumps({"kind": "macos_process_rusage_v4", "pid": args.pid,
            "started_at": time.time(), "duration_limit_s": args.seconds,
            "units": "bytes except pageins/count and proc_start_abstime/nativeclock",
            "lifetime_peak_note": "Includes earlier work by this same process."}) + "\n")
        for count in range(args.seconds):
            try:
                data = first if count == 0 else read(args.pid)
            except OSError as exc:
                file.write(json.dumps({"status": "process_unavailable", "errno": exc.errno}) + "\n")
                break
            if data["proc_start_abstime"] != first["proc_start_abstime"]:
                file.write(json.dumps({"status": "pid_reused"}) + "\n")
                break
            if system is not None:
                try:
                    snapshot = system()
                    data.update(system_available_bytes=snapshot.available,
                                system_pressure=snapshot.pressure,
                                system_total_bytes=snapshot.total)
                except Exception as exc:
                    data["system_telemetry_error"] = type(exc).__name__
            file.write(json.dumps({"elapsed_s": time.monotonic() - started,
                                   "at": time.time(), **data}) + "\n")
            file.flush()
            time.sleep(1)


if __name__ == "__main__":
    main()
