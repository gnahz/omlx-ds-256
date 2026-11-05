#!/usr/bin/env python3
"""External oMLX 0.7.3 recorder, preserving the 0.7.1 archive/rotation code.

No inference, HTTP, MLX imports, guard changes or target-process signals.
Host sampling survives missing PID files; kqueue preserves process exit status.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import importlib.util
import json
import os
from pathlib import Path
import select
import shlex
import subprocess
import sys
import time

from portable import HERE, REPO, state_dir
import flight_recorder_base as base
BASE_PATH = HERE / "flight_recorder_base.py"
OUTPUT = state_dir() / "logs/flight-recorder"
TARGET_DIR = REPO
NOTE_EXITSTATUS = 0x04000000  # Darwin sys/event.h; not exposed by Python's select.
NOTE_EXIT_DETAIL = 0x02000000
NOTE_EXIT_MEMORY = 0x00020000



class ExitWatcher:
    def __init__(self):
        self.queue = select.kqueue()
        self.identities = {}

    def watch(self, pid, start):
        identity = (pid, start)
        if self.identities.get(pid) == identity:
            return
        self.queue.control([select.kevent(
            pid, filter=select.KQ_FILTER_PROC,
            flags=select.KQ_EV_ADD | select.KQ_EV_ENABLE | select.KQ_EV_ONESHOT,
            fflags=select.KQ_NOTE_EXIT | NOTE_EXITSTATUS | NOTE_EXIT_DETAIL,
        )], 0, 0)
        self.identities[pid] = identity

    def poll(self):
        rows = []
        for event in self.queue.control(None, 16, 0):
            if event.flags & select.KQ_EV_ERROR:
                raise OSError(event.data, "kqueue process observation failed")
            if event.filter != select.KQ_FILTER_PROC or not event.fflags & select.KQ_NOTE_EXIT:
                continue
            identity = self.identities.pop(event.ident, (event.ident, None))
            available = bool(event.fflags & NOTE_EXITSTATUS)
            status = event.data if available else None
            rows.append({"pid": identity[0], "proc_start_abstime": identity[1],
                         "observed_at_unix": time.time(), "source": "kqueue_NOTE_EXIT",
                         "exit_status_available": available, "raw_wait_status": status,
                         "exit_code": os.WEXITSTATUS(status) if available and os.WIFEXITED(status) else None,
                         "signal": os.WTERMSIG(status) if available and os.WIFSIGNALED(status) else None,
                         "core_dumped": os.WCOREDUMP(status) if available and os.WIFSIGNALED(status) else False,
                         "kernel_exit_detail_flags": event.fflags,
                         "kernel_memory_exit": bool(event.fflags & NOTE_EXIT_MEMORY),
                         "signal_sender": None,
                         "limitation": "Signal sender and OOM attribution require separate OS evidence."})
        return rows

    def close(self):
        self.queue.close()


class Recorder(base.Recorder):
    def __init__(self, config, *, watcher=None, **kwargs):
        super().__init__(config, **kwargs)
        self.watcher = watcher if watcher is not None else ExitWatcher()
        self.latest = None

    def sample(self):
        for event in self.watcher.poll():
            self._event("target_exit", **event)
        if self.reader is None:
            self.reader, self.native_module = base.load_native_reader()
        error = None
        try:
            pid = self._resolve_pid()
            self._verify_target(pid)
            raw = self.reader(pid)
            if raw.get("process", {}).get("available") is not True:
                raise ProcessLookupError("native process counters unavailable")
            row = base.compact_sample(raw, observed_pid=pid)
            start = row["process"]["proc_start_abstime"]
            self.watcher.watch(pid, start)
            if self.pid != pid or self.proc_start_abstime != start:
                self._verify_target(pid, force=True)
                self._event("target_identity", old_pid=self.pid, pid=pid, proc_start_abstime=start)
                self.pid, self.proc_start_abstime = pid, start
                self._state_dirty = True
            row["target_state"] = "observed"
            row["target_identity_verification_sha256"] = base.canonical_digest(self.target_verification)
        except Exception as exc:
            error = {"error_type": type(exc).__name__, "detail": str(exc)[:512]}
            # Sample host counters through our own PID, but NEVER label the
            # recorder's process footprint as the missing inference process.
            row = base.compact_sample(self.reader(os.getpid()), observed_pid=None)
            row["process"] = {}
            row["target_state"] = "unavailable"
            row["target_error"] = error
            if self.pid is not None:
                self._event("target_unavailable", previous_pid=self.pid, **error)
            self.pid = self.proc_start_abstime = self.target_verification = None
            self._state_dirty = True
            if not isinstance(exc, (FileNotFoundError, ProcessLookupError)):
                self._event_limited("target_observation_error", **error)
        row["service_log"] = base.file_position(self.service_log)
        row["recorder_pid"] = os.getpid()
        self.samples.append(row)
        self._spool(row)
        self.latest = row
        base.atomic_json(self.output_root / "heartbeat.json", {
            "recorder_pid": os.getpid(), "observed_at_unix": row["observed_at_unix"],
            "target_pid": row["pid"], "target_state": row["target_state"],
            "target_error": error, "memory_pressure_raw": row["memory_pressure_raw"],
        })

    def run(self, **kwargs):
        try:
            return super().run(**kwargs)
        finally:
            self.watcher.close()


def make_config():
    state = state_dir()
    return base.Config(archive_root=str(state / "logs/requests"),
        pid_file=str(state / "process.json"), output_root=str(OUTPUT),
        service_log=str(state / "logs/service.log"), api_key_file=str(state / "api-key"), poll_seconds=1.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-config", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()
    config = make_config()
    config.validate()
    if args.check_config:
        print(json.dumps({"status": "passed", "config": asdict(config),
                          "target_contacted": False, "frozen_base_sha256": base.sha256(BASE_PATH)}))
    else:
        Recorder(config).run()


if __name__ == "__main__":
    main()
