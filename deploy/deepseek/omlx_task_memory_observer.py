#!/usr/bin/env python3
"""Observe macOS task memory ledgers and global paging; never control a service.

Only proc_pidinfo/proc_pid_rusage/task_name_for_pid/TASK_VM_INFO metadata is
read. No task read/control port, VM payload, model, inference API, or process
control is used. Evidence is written only to the requested output paths.
"""

import argparse
import ctypes as C
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time


GLOBAL_ACTIVITY = ("compressions", "decompressions", "swapins", "swapouts")
TASK_ACTIVITY = ("compressed_lifetime_bytes", "decompressions", "swapin_bytes")


class BSDInfo(C.Structure):
    _fields_ = [(name, C.c_uint32) for name in (
        "flags", "status", "xstatus", "pid", "ppid", "uid", "gid", "ruid",
        "rgid", "svuid", "svgid", "reserved")]
    _fields_ += [("comm", C.c_char * 16), ("name", C.c_char * 32)]
    _fields_ += [(name, C.c_uint32) for name in (
        "nfiles", "pgid", "pjobc", "tdev", "tpgid")]
    _fields_ += [("nice", C.c_int32), ("start_seconds", C.c_uint64),
                 ("start_microseconds", C.c_uint64)]


class RUsageV4(C.Structure):
    # Stable v4 prefix; room for the entire ABI, not a truncated destination.
    _fields_ = [("uuid", C.c_uint8 * 16)] + [(name, C.c_uint64) for name in (
        "user", "system", "idle_wakeups", "interrupt_wakeups", "pageins",
        "wired", "resident", "phys_footprint", "start_abstime")]
    _fields_ += [("tail", C.c_uint64 * 64)]


def _u64(buffer, offset):
    return C.c_uint64.from_buffer(buffer, offset).value


def decode_task_vm(buffer, words):
    """SDK task_info.h uses pack(4); rev6 swapin ledger ends at byte356."""
    if words * 4 < 356:
        raise ValueError("TASK_VM_INFO reply lacks the rev6 swapin ledger")
    return {
        "vm_info_words": words,
        "page_size": C.c_int32.from_buffer(buffer, 12).value,
        "phys_footprint_bytes": _u64(buffer, 144),
        "resident_bytes": _u64(buffer, 16),
        "compressed_bytes": _u64(buffer, 120),
        "compressed_peak_bytes": _u64(buffer, 128),
        "compressed_lifetime_bytes": _u64(buffer, 136),
        "decompressions": C.c_int32.from_buffer(buffer, 344).value,
        "swapin_bytes": C.c_int64.from_buffer(buffer, 348).value,
    }


def delta(current, previous, fields):
    result = {key: current[key] - previous[key] for key in fields}
    if any(value < 0 for value in result.values()):
        raise ValueError("Cumulative metadata counter decreased")
    return result


def task_deltas(current, previous):
    old = {row["start_id"]: row for row in previous}
    changes = []
    for row in current:
        prior = old.get(row["start_id"])
        if prior is None:
            continue
        values = delta(row, prior, TASK_ACTIVITY)
        if any(values.values()):
            changes.append({"pid": row["pid"], "comm": row["comm"],
                            "start_id": row["start_id"], "delta": values})
    return changes


def strict_global_gate(values):
    return all(values[name] == 0 for name in ("compressions", "swapins", "swapouts"))


class Metadata:
    def __init__(self, uid):
        if sys.platform != "darwin":
            raise RuntimeError("This observer requires macOS")
        self.uid = uid
        self.lib = C.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
        self.proc = C.CDLL("/usr/lib/libproc.dylib", use_errno=True)
        self.lib.mach_task_self.restype = C.c_uint32
        self.lib.mach_host_self.restype = C.c_uint32
        self.lib.task_name_for_pid.argtypes = [C.c_uint32, C.c_int32, C.POINTER(C.c_uint32)]
        self.lib.task_info.argtypes = [C.c_uint32, C.c_uint32, C.c_void_p, C.POINTER(C.c_uint32)]
        self.lib.mach_port_deallocate.argtypes = [C.c_uint32, C.c_uint32]
        self.lib.host_statistics64.argtypes = [C.c_uint32, C.c_int32, C.c_void_p,
                                              C.POINTER(C.c_uint32)]
        self.lib.host_page_size.argtypes = [C.c_uint32, C.POINTER(C.c_size_t)]
        self.lib.sysctlbyname.argtypes = [C.c_char_p, C.c_void_p, C.POINTER(C.c_size_t),
                                         C.c_void_p, C.c_size_t]
        self.proc.proc_listpids.argtypes = [C.c_uint32, C.c_uint32, C.c_void_p, C.c_int]
        self.proc.proc_pidinfo.argtypes = [C.c_int, C.c_int, C.c_uint64, C.c_void_p, C.c_int]
        self.proc.proc_pid_rusage.argtypes = [C.c_int, C.c_int, C.c_void_p]
        self.me, self.host = self.lib.mach_task_self(), self.lib.mach_host_self()
        page = C.c_size_t()
        if self.lib.host_page_size(self.host, C.byref(page)) or page.value <= 0:
            raise RuntimeError("Cannot read host page size")
        self.page_size = page.value

    def global_vm(self):
        buffer = (C.c_int * 256)()
        words = C.c_uint32(104)  # SDK rev5 vm_statistics64_data_t,416 bytes.
        rc = self.lib.host_statistics64(self.host, 4, buffer, C.byref(words))
        if rc == -307 and 36 <= words.value <= 256:
            rc = self.lib.host_statistics64(self.host, 4, buffer, C.byref(words))
        if rc or not 36 <= words.value <= 256:
            raise RuntimeError(f"host_statistics64 failed: rc={rc},words={words.value}")
        size = words.value * 4
        page = self.page_size
        free = C.c_uint32.from_buffer(buffer, 0).value * page
        speculative = C.c_uint32.from_buffer(buffer, 92).value * page
        result = {"at_utc": datetime.now(timezone.utc).isoformat(), "at_unix": time.time(),
                  "vm_info_words": words.value, "page_size": page,
                  "free_including_speculative_bytes": free,
                  "speculative_bytes": speculative,
                  "immediately_free_bytes": max(0, free - speculative)}
        for name, offset in (("active_bytes", 4), ("inactive_bytes", 8), ("wired_bytes", 12),
                             ("compressor_physical_bytes", 128),
                             ("external_total_bytes", 136), ("internal_bytes", 140)):
            result[name] = C.c_uint32.from_buffer(buffer, offset).value * page
        for name, offset in (("decompressions", 96), ("compressions", 104),
                             ("swapins", 112), ("swapouts", 120)):
            result[name] = _u64(buffer, offset)
        if size >= 344:
            result["active_external_bytes"] = _u64(buffer, 328) * page
            result["inactive_external_bytes"] = _u64(buffer, 336) * page
        return result

    def bsd(self, pid):
        info = BSDInfo()
        amount = self.proc.proc_pidinfo(pid, 3, 0, C.byref(info), C.sizeof(info))
        if amount != C.sizeof(info):
            raise OSError(C.get_errno(), "proc_pidinfo unavailable or process exited")
        return info

    def pids(self):
        amount = self.proc.proc_listpids(1, 0, None, 0)
        if amount <= 0:
            raise RuntimeError("Cannot enumerate process metadata")
        values = (C.c_int * (amount // 4 + 1024))()
        copied = self.proc.proc_listpids(1, 0, values, C.sizeof(values))
        if copied <= 0 or copied > C.sizeof(values):
            raise RuntimeError("Cannot enumerate bounded process metadata")
        return sorted(pid for pid in values[:copied // 4] if pid > 0)

    def tasks(self):
        rows, errors = [], []
        for pid in self.pids():
            try:
                info = self.bsd(pid)
            except OSError:
                continue  # May have exited before its owner can be checked.
            if info.uid != self.uid:
                continue
            try:
                usage = RUsageV4()
                if self.proc.proc_pid_rusage(pid, 4, C.byref(usage)):
                    raise OSError(C.get_errno(), "proc_pid_rusage unavailable")
                port = C.c_uint32()
                rc = self.lib.task_name_for_pid(self.me, pid, C.byref(port))
                if rc:
                    raise OSError(rc, "task_name_for_pid denied")
                try:
                    buffer, words = (C.c_int * 128)(), C.c_uint32(93)
                    rc = self.lib.task_info(port.value, 22, buffer, C.byref(words))
                    if rc:
                        raise OSError(rc, "TASK_VM_INFO denied")
                    values = decode_task_vm(buffer, words.value)
                finally:
                    self.lib.mach_port_deallocate(self.me, port.value)
                after = self.bsd(pid)
                start = (info.start_seconds, info.start_microseconds)
                if start != (after.start_seconds, after.start_microseconds) or after.uid != self.uid:
                    raise ValueError("PID identity changed while observing metadata")
                uuid = bytes(usage.uuid).hex()
                rows.append({"pid": pid, "ppid": info.ppid, "uid": info.uid,
                             "comm": (info.name or info.comm).decode(errors="replace"),
                             "start_unix_seconds": info.start_seconds,
                             "start_microseconds": info.start_microseconds,
                             "start_abstime": usage.start_abstime, "executable_uuid": uuid,
                             "start_id": f"{pid}:{start[0]}:{start[1]}:{usage.start_abstime}",
                             **values})
            except (OSError, ValueError) as error:
                errors.append({"pid": pid, "comm": (info.name or info.comm).decode(errors="replace"),
                               "error": str(error)})
        return rows, errors


def service_pid(path):
    if path is None:
        return None
    try:
        # The existing public lifecycle writes one decimal PID. Bound the read.
        with Path(path).open("rt") as stream:
            raw = stream.read(32).strip()
        return int(raw) if raw.isdecimal() and int(raw) > 0 else None
    except OSError:
        return None


def observe(provider, *, duration, interval, output, pid_file=None, summary_path=None):
    start, previous, baseline = time.monotonic(), None, None
    global_events, task_events, failures = [], [], []
    last_tasks, last_global, count, latest_service = [], None, 0, None
    # Fail on an accidental overwrite of earlier qualification evidence.
    with Path(output).open("x") as stream:
        while True:
            scan_start = time.monotonic()
            before = provider.global_vm()
            tasks, errors = provider.tasks()
            after = provider.global_vm()
            if baseline is None:
                baseline = before
            activity = delta(after, baseline, GLOBAL_ACTIVITY)
            incremental = delta(after, previous["global"], GLOBAL_ACTIVITY) if previous else delta(after, before, GLOBAL_ACTIVITY)
            changes = task_deltas(tasks, previous["tasks"]) if previous else []
            pid = service_pid(pid_file)
            service = next((row for row in tasks if row["pid"] == pid), None)
            if service is not None:
                latest_service = service
            row = {"sample": count, "elapsed_s": time.monotonic() - start,
                   "scan_seconds": time.monotonic() - scan_start,
                   "global_before_tasks": before, "global_after_tasks": after,
                   "global_delta_from_baseline": activity,
                   "global_delta_since_previous": incremental,
                   "tasks": tasks, "task_deltas_since_previous": changes,
                   "task_access_errors": errors, "service": service}
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
            stream.flush()
            if any(incremental[name] for name in ("compressions", "swapins", "swapouts")):
                global_events.append({"sample": count, "elapsed_s": row["elapsed_s"],
                                      "delta": incremental, "same_window_task_changes": changes})
            task_events.extend({"sample": count, "elapsed_s": row["elapsed_s"], **item} for item in changes)
            failures.extend(errors)
            count += 1
            previous, last_tasks, last_global = {"global": after, "tasks": tasks}, tasks, after
            remaining = duration - (time.monotonic() - start)
            if remaining <= 0:
                break
            time.sleep(min(remaining, max(0, interval - (time.monotonic() - scan_start))))
    final_delta = delta(last_global, baseline, GLOBAL_ACTIVITY)
    summary = {"metadata_only": True, "observer_pid": os.getpid(), "uid": provider.uid,
               "duration_seconds": time.monotonic() - start, "samples": count,
               "baseline_global": baseline, "final_global": last_global,
               "global_delta": final_delta, "strict_global_gate_passed": strict_global_gate(final_delta),
               "global_paging_events": global_events, "task_activity_events": task_events,
               "latest_service": latest_service, "last_readable_task_count": len(last_tasks),
               "task_access_error_count": len(failures),
               "attribution_limit": "Task scans are sequential; matching interval deltas are candidates, not proof of exclusive causation. Unreadable/exited tasks and activity before observation remain unattributed."}
    if summary_path is not None:
        with Path(summary_path).open("x") as stream:
            json.dump(summary, stream, indent=2)
            stream.write("\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--duration", type=float, default=20)
    parser.add_argument("--interval", type=float, default=.5)
    parser.add_argument("--uid", type=int, default=os.getuid())
    parser.add_argument("--pid-file", type=Path)
    args = parser.parse_args()
    if not 0 < args.duration <= 3600 or not .5 <= args.interval <= 1 or args.uid < 0:
        parser.error("duration must be(0,3600], interval[.5,1], uid nonnegative")
    try:
        summary = observe(Metadata(args.uid), duration=args.duration, interval=args.interval,
                          output=args.output, pid_file=args.pid_file, summary_path=args.summary)
    except (OSError, RuntimeError, ValueError) as error:
        parser.exit(1, f"Observation incomplete; no passing memory gate established: {error}\n")
    print(json.dumps({key: summary[key] for key in ("samples", "global_delta", "strict_global_gate_passed",
                                                  "task_access_error_count", "latest_service")}))


if __name__ == "__main__":
    main()
