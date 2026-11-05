#!/usr/bin/env python3
"""Independent launchd lifecycle for the 073 memory recorder, never oMLX."""
import argparse
import hashlib
import sys
import fcntl
import json
import os
from pathlib import Path
import plistlib
import subprocess
import time

from portable import HERE, REPO, state_dir
ROOT = REPO
OUT = state_dir() / "logs/flight-recorder"
RECORDER = HERE / "recorder.py"
LABEL = "com.omlx-ds-256.recorder." + hashlib.sha256(str(state_dir()).encode()).hexdigest()[:12]
DOMAIN = f"user/{os.getuid()}"
JOB = f"{DOMAIN}/{LABEL}"
PLIST = Path.home() / "Library/LaunchAgents" / f"{LABEL}.plist"


def run(*args):
    return subprocess.run(args, text=True, capture_output=True, check=False, timeout=20)


def status():
    result = run("/bin/launchctl", "print", JOB)
    pid = None
    if result.returncode == 0:
        for line in result.stdout.splitlines():
            if line.strip().startswith("pid = "):
                pid = int(line.strip().split("=", 1)[1])
                break
    heartbeat = {}
    try:
        heartbeat = json.loads((OUT / "heartbeat.json").read_text())
    except (OSError, ValueError):
        pass
    verified = False
    if pid:
        command = run("/bin/ps", "-ww", "-p", str(pid), "-o", "uid=,command=")
        if command.returncode == 0:
            owner, text = command.stdout.strip().split(None, 1)
            verified = int(owner) == os.getuid() and str(RECORDER) in text and "--run" in text
    ready = bool(verified and heartbeat.get("recorder_pid") == pid
                 and 0 <= time.time() - heartbeat.get("observed_at_unix", 0) < 5)
    return {"label": LABEL, "loaded": result.returncode == 0, "pid": pid,
            "identity_verified": verified, "ready": ready,
            "heartbeat": heartbeat, "logs": str(OUT)}


def plist_data():
    return {"Label": LABEL,
            "ProgramArguments": [sys.executable, "-u", str(RECORDER), "--run"],
            "WorkingDirectory": str(ROOT), "RunAtLoad": True, "KeepAlive": True,
            "ThrottleInterval": 10, "ProcessType": "Background", "Nice": 10,
            "LowPriorityIO": True, "EnablePressuredExit": False, "ExitTimeOut": 30,
            "Umask": 63, "EnvironmentVariables": {"PATH": "/usr/bin:/bin:/usr/sbin:/sbin", "PYTHONNOUSERSITE": "1", "OMLX_DS_STATE": str(state_dir())},
            "StandardOutPath": str(OUT / "launchd.stdout.log"),
            "StandardErrorPath": str(OUT / "launchd.stderr.log")}


def start():
    previous = status()
    if previous["ready"]:
        return {**previous, "already_running": True}
    if previous["loaded"]:
        raise RuntimeError(f"recorder job loaded but unhealthy; use restart; logs: {OUT}")
    PLIST.parent.mkdir(parents=True, exist_ok=True)
    if PLIST.is_symlink():
        raise RuntimeError("refusing symlink launch agent plist")
    temporary = PLIST.with_suffix(".plist.tmp")
    with temporary.open("wb") as stream:
        plistlib.dump(plist_data(), stream)
    temporary.chmod(0o600)
    temporary.replace(PLIST)
    result = run("/bin/launchctl", "enable", JOB)
    if result.returncode:
        raise RuntimeError(f"recorder enable failed: {result.stderr.strip()}")
    result = run("/bin/launchctl", "bootstrap", DOMAIN, str(PLIST))
    if result.returncode:
        raise RuntimeError(f"recorder bootstrap failed: {result.stderr.strip()}; logs: {OUT}")
    for _ in range(40):
        current = status()
        if current["ready"]:
            return current
        time.sleep(0.25)
    raise RuntimeError(f"recorder did not publish a fresh heartbeat; logs: {OUT}")


def stop():
    previous = status()
    result = run("/bin/launchctl", "disable", JOB)
    if result.returncode:
        raise RuntimeError(f"recorder disable failed: {result.stderr.strip()}")
    if not previous["loaded"]:
        return {**previous, "already_stopped": True}
    if previous["pid"] and not previous["identity_verified"]:
        raise RuntimeError("refusing to stop a recorder with unverified process identity")
    result = run("/bin/launchctl", "bootout", JOB)
    if result.returncode:
        raise RuntimeError(f"recorder bootout failed: {result.stderr.strip()}")
    for _ in range(120):
        try:
            if previous["pid"]:
                os.kill(previous["pid"], 0)
            else:
                return {**status(), "stopped_pid": None}
        except ProcessLookupError:
            return {**status(), "stopped_pid": previous["pid"]}
        time.sleep(0.25)
    raise RuntimeError("recorder termination was not confirmed after 30 seconds")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("start", "stop", "restart", "status"))
    args = parser.parse_args()
    OUT.mkdir(mode=0o700, parents=True, exist_ok=True)
    if OUT.is_symlink() or OUT.stat().st_uid != os.getuid():
        raise RuntimeError("recorder output directory must be owned and not a symlink")
    OUT.chmod(0o700)
    lock = os.open(OUT / "control.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if args.action == "restart":
            stop()
            result = start()
        else:
            result = {"start": start, "stop": stop, "status": status}[args.action]()
        print(json.dumps(result, indent=2))
    finally:
        os.close(lock)


if __name__ == "__main__":
    main()
