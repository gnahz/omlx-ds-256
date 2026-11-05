#!/usr/bin/env python3
"""Download the pinned 1017 checkpoint and verify its published SHA256SUMS.

This is a file-management tool. It never imports MLX or starts a service.
Verification streams files through a bounded buffer; it does not load a model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
RECEIPT = ".omlx-ds-256-verification.json"
CHUNK_BYTES = 8 * 1024**2


def load_manifest():
    return json.loads((HERE / "model-manifest.json").read_text())


def identity(stat):
    # Hardlinking the verified overlay changes ctime, but not file contents or
    # mtime. The runtime takes its own stronger identity snapshot when loading.
    return {"device": stat.st_dev, "inode": stat.st_ino,
            "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def within(root, name):
    relative = PurePosixPath(name)
    if (not name or relative.is_absolute() or ".." in relative.parts
            or "\\" in name or str(relative) != name):
        raise ValueError(f"Invalid published filename: {name!r}")
    path = root.joinpath(*relative.parts)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"File escapes model directory: {name}")
    return path


def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        before = identity(os.fstat(stream.fileno()))
        if sys.platform == "darwin":
            import fcntl
            fcntl.fcntl(stream.fileno(), fcntl.F_NOCACHE, 1)
        for block in iter(lambda: stream.read(CHUNK_BYTES), b""):
            sha.update(block)
        if before != identity(os.fstat(stream.fileno())):
            raise ValueError(f"File changed while verifying: {path.name}")
    if before != identity(path.stat()):
        raise ValueError(f"File was replaced while verifying: {path.name}")
    return sha.hexdigest(), before


def checksums(model_dir, manifest):
    path = model_dir / "SHA256SUMS"
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != manifest["sha256"]["SHA256SUMS"]:
        raise ValueError("SHA256SUMS does not match the pinned release")
    expected = {}
    for line in data.decode("utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if not match:
            raise ValueError("Malformed published SHA256SUMS line")
        checksum, name = match.groups()
        within(model_dir, name)
        if name in expected:
            raise ValueError(f"Duplicate published filename: {name}")
        expected[name] = checksum
    expected["SHA256SUMS"] = manifest["sha256"]["SHA256SUMS"]
    for name, checksum in manifest["sha256"].items():
        if expected.get(name) != checksum:
            raise ValueError(f"Release checksum disagrees with pinned manifest: {name}")
    return expected


def save_json(path, value):
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def verify(model_dir, manifest):
    expected = checksums(model_dir, manifest)
    files = {}
    total = len(expected)
    for count, (name, wanted) in enumerate(expected.items(), 1):
        actual, stat = digest(within(model_dir, name))
        if actual != wanted:
            raise ValueError(f"Checksum mismatch: {name}")
        files[name] = {"sha256": actual, **stat}
        print(f"Verified {count}/{total}: {name}", flush=True)
    result = {"schema_version": 1, "repo_id": manifest["repo_id"],
              "revision": manifest["revision"], "files": files}
    save_json(model_dir / RECEIPT, result)
    return result


def verify_receipt(model_dir, manifest):
    """Check that all previously hashed files retain their local identities."""
    expected = checksums(model_dir, manifest)
    path = model_dir / RECEIPT
    if not path.is_file():
        raise ValueError("Run download_model.py --verify-only before preparing metadata")
    receipt = json.loads(path.read_text())
    if (receipt.get("schema_version") != 1
            or receipt.get("repo_id") != manifest["repo_id"]
            or receipt.get("revision") != manifest["revision"]
            or set(receipt.get("files", {})) != set(expected)):
        raise ValueError("Verification receipt belongs to another release")
    for name, wanted in expected.items():
        record = receipt["files"][name]
        if (record.get("sha256") != wanted
                or any(record.get(k) != v
                       for k, v in identity(within(model_dir, name).stat()).items())):
            raise ValueError(f"File changed since verification; re-run --verify-only: {name}")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--max-workers", type=int, default=2)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--dry-run", action="store_true",
                        help="Ask hf for the download plan; download no weights")
    action.add_argument("--verify-only", action="store_true",
                        help="Hash an existing download without network access")
    args = parser.parse_args()
    model_dir = args.model_dir.expanduser().resolve()
    if model_dir.is_relative_to(HERE.parents[1]):
        parser.error("Keep model weights outside this source repository")
    if args.max_workers < 1:
        parser.error("--max-workers must be positive")
    manifest = load_manifest()
    if not args.verify_only:
        hf = shutil.which("hf")
        if hf is None:
            parser.error("Install the Hugging Face CLI first: python -m pip install -U huggingface_hub")
        command = [hf, "download", manifest["repo_id"], "--revision", manifest["revision"],
                   "--local-dir", str(model_dir), "--max-workers", str(args.max_workers)]
        if args.dry_run:
            command.append("--dry-run")
        print(f"Pinned model: {manifest['repo_id']} @ {manifest['revision']}", flush=True)
        print(f"Complete download: {manifest['total_download_bytes'] / 1024**3:.2f} GiB; "
              "reserve additional space for builds and caches.", flush=True)
        subprocess.run(command, check=True)
        if args.dry_run:
            return
    result = verify(model_dir, manifest)
    print(f"Verified {len(result['files'])} files. Next run prepare_metadata.py.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        raise SystemExit(str(exc)) from exc
