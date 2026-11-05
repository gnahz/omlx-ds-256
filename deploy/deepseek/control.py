#!/usr/bin/env python3
"""Manual lifecycle controls; never discover, stop, or replace another service."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import secrets
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

from portable import HERE, REPO, atomic_json, configure, profile_path, state_dir, validate_configuration


def identity(pid):
    import psutil
    try:
        process = psutil.Process(pid)
        if process.status() == psutil.STATUS_ZOMBIE:
            return None
        return {"pid": pid, "created": process.create_time(), "executable": process.exe()}
    except (psutil.NoSuchProcess, psutil.ZombieProcess):
        return None


def owns(record):
    if not record or record.get("source") != str(REPO) or not record.get("token"):
        return False
    current = identity(record["pid"])
    return current is not None and all(current[name] == record.get(name) for name in current)


def read_record():
    path = state_dir() / "process.json"
    return json.loads(path.read_text()) if path.exists() else None


@contextmanager
def lifecycle_lock():
    state_dir().mkdir(parents=True, exist_ok=True, mode=0o700)
    path = state_dir() / "lifecycle.lock"
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    with os.fdopen(fd, "a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another lifecycle command is still running") from None
        yield


def check_port_available(host, port):
    # Bind only; never send a request to an unowned listener.
    addresses = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    for family, kind, protocol, _, address in addresses:
        with socket.socket(family, kind, protocol) as sock:
            try:
                sock.bind(address)
            except OSError as error:
                raise RuntimeError(f"Port {port} is unavailable; no other service was changed") from error


def ready(profile, record):
    if not owns(record):
        return False
    host = profile["host"]
    host = "127.0.0.1" if host == "0.0.0.0" else "::1" if host == "::" else host
    host = f"[{host}]" if ":" in host else host
    request = urllib.request.Request(f"http://{host}:{profile['port']}/api/runtime-profile",
        headers={"Authorization": "Bearer " + Path(profile["api_key_file"]).read_text().strip()})
    # Local service health is never routed via a proxy environment variable.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=2) as response:
            value = json.load(response)
    except (OSError, ValueError, urllib.error.URLError):
        return False
    return (value.get("ready") is True and not value.get("error")
            and value.get("pid") == record["pid"]
            and value.get("instance_token") == record["token"]
            and value.get("launch_identity", {}).get("source") == str(REPO))


def stop(timeout=180):
    record = read_record()
    if not owns(record):
        # A stale PID may have been reused. Never signal it.
        if record and identity(record["pid"]) is not None:
            raise RuntimeError("Stored PID identity differs; refusing to signal that process")
        (state_dir() / "process.json").unlink(missing_ok=True)
        print("Stopped (no owned service process).")
        return
    os.kill(record["pid"], signal.SIGTERM)
    deadline = time.monotonic() + timeout
    while owns(record) and time.monotonic() < deadline:
        time.sleep(.25)
    if owns(record):
        raise RuntimeError("Graceful stop timed out; process retained. Inspect logs before retrying.")
    (state_dir() / "process.json").unlink(missing_ok=True)
    print("Stopped; service process has exited and its address space is released.")


def start():
    record = read_record()
    if owns(record):
        print(f"Already running: PID {record['pid']}; logs: {state_dir() / 'logs/service.log'}")
        return
    if record and identity(record["pid"]) is not None:
        raise RuntimeError("Stored PID belongs to another process; refusing to overwrite its record")
    profile = json.loads(profile_path().read_text())
    validate_configuration(profile)
    check_port_available(profile["host"], profile["port"])
    token = secrets.token_hex(32)
    env = dict(os.environ, OMLX_DS_STATE=str(state_dir()), OMLX_DS_INSTANCE_TOKEN=token,
               PYTHONUNBUFFERED="1")
    logs = state_dir() / "logs"
    logs.mkdir(mode=0o700, exist_ok=True)
    log_path = logs / "service.log"
    fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    with os.fdopen(fd, "ab", buffering=0) as log:
        child = subprocess.Popen([profile["python"], str(HERE / "launch.py"), "serve"],
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            cwd=REPO, env=env, start_new_session=True, close_fds=True)
    current = identity(child.pid)
    if current is None:
        raise RuntimeError(f"Service exited immediately. See {log_path}")
    record = dict(current, source=str(REPO), token=token)
    atomic_json(state_dir() / "process.json", record)
    # This assertion is owned by and automatically ends with the service PID.
    # It does not change global power settings or defeat thermal protection.
    if Path("/usr/bin/caffeinate").is_file():
        subprocess.Popen(["/usr/bin/caffeinate", "-i", "-w", str(child.pid)],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True, close_fds=True)
    print(f"Starting PID {child.pid}; logs: {log_path}", flush=True)
    deadline = time.monotonic() + profile["startup_timeout_seconds"]
    while time.monotonic() < deadline:
        if not owns(record):
            raise RuntimeError(f"Service exited before readiness. See {log_path}")
        if ready(profile, record):
            print(f"Ready: {profile['host']}:{profile['port']} (PID {child.pid})")
            return
        time.sleep(1)
    # Stop only the matching child we created; never touch another listener.
    stop()
    raise RuntimeError(f"Startup readiness timed out. See {log_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    setup = sub.add_parser("configure")
    setup.add_argument("--model-path", required=True)
    setup.add_argument("--metadata-proof", required=True)
    setup.add_argument("--host", default="127.0.0.1")
    setup.add_argument("--port", type=int, default=8080)
    for action in ("start", "stop", "restart", "status"):
        sub.add_parser(action)
    args = parser.parse_args()
    try:
        with lifecycle_lock():
            if args.action == "configure":
                if owns(read_record()):
                    raise RuntimeError("Stop this deployment before changing its configuration")
                configure(args.model_path, args.metadata_proof, host=args.host, port=args.port)
                print(f"Configuration: {profile_path()}\nPrivate API key: {state_dir() / 'api-key'}")
            elif args.action == "start":
                start()
            elif args.action == "stop":
                stop()
            elif args.action == "restart":
                stop()
                start()
            else:
                record = read_record()
                print(json.dumps({"running": owns(record), "pid": record.get("pid") if record else None,
                                  "state": str(state_dir()), "log": str(state_dir() / "logs/service.log")}, indent=2))
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        parser.exit(1, str(error) + "\n")


if __name__ == "__main__":
    main()
