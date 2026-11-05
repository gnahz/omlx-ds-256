#!/usr/bin/env python3
"""Prepare local identities for the already published exact-metadata overlay.

Hardlinks preserve the published bytes without copying the 7.92 GiB overlay.
The output must be outside the model directory on the same filesystem. This
tool reads bounded safetensors headers only, never imports MLX, never changes
the downloaded checkpoint, and never starts a service.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import struct

from download_model import identity, load_manifest, verify_receipt

SIDECAR = "DeepSeek-V4.1-LSQ3-G128-exact-metadata.json"
PROOF = "metadata-proof.json"
MAX_HEADER_BYTES = 1024**2


def pinned_json(path, expected):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(f"Pinned document checksum mismatch: {path.name}")
    return json.loads(data), data


def header_identity(path, pin):
    with path.open("rb") as stream:
        before = identity(os.fstat(stream.fileno()))
        raw = stream.read(8)
        if len(raw) != 8:
            raise ValueError(f"Truncated safetensors header: {path.name}")
        length = struct.unpack("<Q", raw)[0]
        if not 2 <= length <= min(MAX_HEADER_BYTES, before["size"] - 8):
            raise ValueError(f"Invalid safetensors header length: {path.name}")
        raw += stream.read(length)
        if (len(raw) != length + 8 or len(raw) != pin.get("header_bytes")
                or before["size"] != pin.get("size")
                or hashlib.sha256(raw).hexdigest() != pin.get("header_sha256")
                or before != identity(os.fstat(stream.fileno()))):
            raise ValueError(f"Pinned safetensors header mismatch: {path.name}")
    return before


def prepare(model_dir, output, manifest):
    model_dir = model_dir.expanduser().resolve()
    output = output.expanduser().resolve()
    if output == model_dir or output.is_relative_to(model_dir):
        raise ValueError("Metadata output must be outside the model directory")
    if output.exists():
        raise ValueError("Output already exists; choose a new metadata directory")
    if not output.parent.is_dir():
        raise ValueError("Create the output's parent directory first")
    verify_receipt(model_dir, manifest)
    original = model_dir / "exact-metadata"
    sidecar, _ = pinned_json(original / SIDECAR, manifest["sha256"]["exact-metadata/" + SIDECAR])
    proof, proof_bytes = pinned_json(original / PROOF, manifest["sha256"]["exact-metadata/" + PROOF])
    config, _ = pinned_json(model_dir / "config.json", manifest["sha256"]["config.json"])
    pinned_json(model_dir / "model.safetensors.index.json",
                manifest["sha256"]["model.safetensors.index.json"])
    custom = config.get("omlx_deepseek_v41") or {}
    if (config.get("model_type") != manifest["required_model_type"]
            or custom.get("attention_output_quantization") is not True
            or custom.get("preserve_mtp") is not True):
        raise ValueError("Model is missing the required 1017 configuration")
    count = manifest["exact_metadata_tensor_files"]
    projections = proof.get("projections") or []
    encoded = {record.get("file") for record in projections}
    if (proof.get("status") != "passed" or proof.get("quantization_codes_changed") is not False
            or proof.get("original_tensors_retained") is not True
            or proof.get("exact_dictionary_projections") != count
            or len(projections) != count or len(encoded) != count
            or proof.get("saved_resident_bytes") != manifest["exact_metadata_saved_resident_bytes"]
            or sidecar.get("requantized") is not False
            or any(sidecar.get("source_" + name + "_sha256") != manifest["sha256"][filename]
                   for name, filename in (("config", "config.json"), ("index", "model.safetensors.index.json")))):
        raise ValueError("Invalid or mismatched published exact-metadata proof")
    if any(not isinstance(name, str) or Path(name).name != name for name in encoded):
        raise ValueError("Invalid encoded metadata filename")
    pins = sidecar.get("files") or []
    if len(pins) != manifest["checkpoint_tensor_files"] + count:
        raise ValueError("The sidecar does not pin every checkpoint and overlay file")
    links, prepared, seen = [], [], set()
    for pin in pins:
        # Published sidecars intentionally refer to checkpoint shards using ../.
        # Resolve, then enforce containment instead of accepting arbitrary paths.
        source = (original / pin["path"]).resolve()
        if not source.is_relative_to(model_dir) or source in seen:
            raise ValueError("Duplicate source or sidecar path outside checkpoint")
        seen.add(source)
        stat = header_identity(source, pin)
        is_encoded = source.parent == original and source.name in encoded
        if is_encoded:
            if stat["device"] != output.parent.stat().st_dev:
                raise ValueError("Place output next to the model on the same filesystem for hardlinks")
            links.append((source, source.name, stat))
        elif source.parent == original:
            raise ValueError("Unrecognized file in exact-metadata directory")
        prepared.append({**pin, "mtime_ns": stat["mtime_ns"],
                         "path": source.name if is_encoded else os.path.relpath(source, output)})
    if {name for _, name, _ in links} != encoded:
        raise ValueError("The sidecar is missing encoded files")
    # Detect a checkpoint changed during header inspection, before any writes.
    verify_receipt(model_dir, manifest)
    output.mkdir(mode=0o700, exist_ok=False)
    for source, name, stat in links:
        os.link(source, output / name)
        if stat != identity((output / name).stat()):
            raise ValueError("Metadata changed while linking; discard incomplete output")
    (output / PROOF).write_bytes(proof_bytes)
    sidecar["files"] = prepared
    (output / SIDECAR).write_text(json.dumps(sidecar, indent=2) + "\n")
    return output / PROOF


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="New sibling directory, e.g. /models/DeepSeek-1017-exact-metadata")
    args = parser.parse_args()
    model_dir = args.model_dir.expanduser().resolve()
    output = args.output.expanduser().resolve()
    proof = prepare(model_dir, output, load_manifest())
    print(f"Prepared 120 exact metadata projections without copying payloads.\nMetadata proof: {proof}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError) as exc:
        raise SystemExit(str(exc)) from exc
