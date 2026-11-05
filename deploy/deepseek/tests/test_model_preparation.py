"""Tiny CPU fixtures for the pinned model download and overlay preparation.

No model, MLX, network, or running service is needed. The fixture deliberately
uses one tiny projection to exercise the same assembly and integrity checks as
the public 120-projection package.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import struct
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import download_model as download
import prepare_metadata as metadata


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    data = (json.dumps(value) + "\n").encode()
    path.write_bytes(data)
    return data


def pin_fixture(model, manifest):
    """Make a complete small release, including its independently pinned sums."""
    hashes = {
        str(path.relative_to(model)): sha(path.read_bytes())
        for path in sorted(model.rglob("*"))
        if path.is_file() and path.name not in ("SHA256SUMS", download.RECEIPT)
    }
    sums = "".join(f"{digest}  {name}\n" for name, digest in hashes.items()).encode()
    (model / "SHA256SUMS").write_bytes(sums)
    hashes["SHA256SUMS"] = sha(sums)
    manifest["sha256"] = hashes
    download.verify(model, manifest)


@pytest.fixture
def package(tmp_path, monkeypatch):
    def no_process(*args, **kwargs):
        pytest.fail("Model preparation must not launch a download or a service")

    monkeypatch.setattr(download.subprocess, "run", no_process)
    model = (tmp_path / "model").resolve()
    model.mkdir()
    overlay = model / "exact-metadata"
    overlay.mkdir()
    header = b'{"x":{"dtype":"U8","shape":[4],"data_offsets":[0,4]}}'
    raw = struct.pack("<Q", len(header)) + header + b"1234"
    (model / "weight.safetensors").write_bytes(raw)
    (overlay / "layer-00-w1.safetensors").write_bytes(raw)
    config = write_json(model / "config.json", {
        "model_type": "deepseek_v41",
        "omlx_deepseek_v41": {
            "preserve_mtp": True, "attention_output_quantization": True,
        },
    })
    index = write_json(model / "model.safetensors.index.json", {"weight_map": {}})
    write_json(overlay / metadata.PROOF, {
        "status": "passed", "quantization_codes_changed": False,
        "original_tensors_retained": True, "exact_dictionary_projections": 1,
        "saved_resident_bytes": 16,
        "projections": [{"file": "layer-00-w1.safetensors"}],
    })
    header_pin = {
        "size": len(raw), "header_bytes": 8 + len(header),
        "header_sha256": sha(raw[:-4]),
    }
    sidecar = {
        "requantized": False, "source_config_sha256": sha(config),
        "source_index_sha256": sha(index),
        "files": [
            dict(header_pin, path="../weight.safetensors"),
            dict(header_pin, path="layer-00-w1.safetensors"),
        ],
    }
    write_json(overlay / metadata.SIDECAR, sidecar)
    manifest = {
        "repo_id": "fixture/model", "revision": "0" * 40,
        "required_model_type": "deepseek_v41", "checkpoint_tensor_files": 1,
        "exact_metadata_tensor_files": 1, "exact_metadata_saved_resident_bytes": 16,
    }
    pin_fixture(model, manifest)
    return SimpleNamespace(
        model=model, overlay=overlay, output=tmp_path / "prepared",
        manifest=manifest, sidecar=sidecar, header_pin=header_pin,
    )


def test_preparation_preserves_source_and_hardlinks_verified_overlay(package):
    p = package
    before = {path: path.read_bytes() for path in p.model.rglob("*") if path.is_file()}
    proof_path = metadata.prepare(p.model, p.output, p.manifest)
    assert proof_path.read_bytes() == (p.overlay / metadata.PROOF).read_bytes()
    assert os.path.samefile(p.output / "layer-00-w1.safetensors",
                           p.overlay / "layer-00-w1.safetensors")
    sidecar = json.loads((p.output / metadata.SIDECAR).read_text())
    for pin in sidecar["files"]:
        source = (p.output / pin["path"]).resolve()
        assert pin["mtime_ns"] == source.stat().st_mtime_ns
        assert pin["size"] == source.stat().st_size
    assert all(path.read_bytes() == contents for path, contents in before.items())
    # Linking changes ctime/link count, but must preserve the content receipt.
    download.verify_receipt(p.model, p.manifest)


def test_full_verification_detects_payload_corruption_even_with_same_mtime(package):
    path = package.model / "weight.safetensors"
    before = path.stat()
    path.write_bytes(path.read_bytes()[:-1] + b"X")
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    with pytest.raises(ValueError, match="Checksum mismatch: weight.safetensors"):
        download.verify(package.model, package.manifest)


def test_receipt_detects_mtime_drift_before_any_output_is_written(package):
    path = package.model / "weight.safetensors"
    before = path.stat()
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000))
    with pytest.raises(ValueError, match="changed since verification"):
        metadata.prepare(package.model, package.output, package.manifest)
    assert not package.output.exists()


def test_receipt_detects_replacement_even_with_identical_size_and_mtime(package):
    path = package.model / "weight.safetensors"
    before = path.stat()
    replacement = path.with_suffix(".replacement")
    replacement.write_bytes(path.read_bytes())
    os.utime(replacement, ns=(before.st_atime_ns, before.st_mtime_ns))
    os.replace(replacement, path)
    with pytest.raises(ValueError, match="changed since verification"):
        download.verify_receipt(package.model, package.manifest)


def test_preparation_requires_full_verification_receipt(package):
    (package.model / download.RECEIPT).unlink()
    with pytest.raises(ValueError, match="--verify-only"):
        metadata.prepare(package.model, package.output, package.manifest)
    assert not package.output.exists()


def test_receipt_from_different_revision_is_rejected(package):
    path = package.model / download.RECEIPT
    receipt = json.loads(path.read_text())
    receipt["revision"] = "f" * 40
    write_json(path, receipt)
    with pytest.raises(ValueError, match="another release"):
        download.verify_receipt(package.model, package.manifest)


def test_checksum_list_itself_must_match_manifest(package):
    path = package.model / "SHA256SUMS"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="SHA256SUMS does not match"):
        download.verify(package.model, package.manifest)


@pytest.mark.parametrize("name", ["../escape", "/absolute", "a/../escape", "a\\escape"])
def test_checksum_paths_cannot_escape_model_directory(package, name):
    with pytest.raises(ValueError, match="Invalid published filename"):
        download.within(package.model, name)


def test_checksum_paths_cannot_escape_through_symlinks(package):
    external = package.model.parent / "external.safetensors"
    path = package.model / "weight.safetensors"
    external.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(external)
    with pytest.raises(ValueError, match="escapes model directory"):
        download.checksums(package.model, package.manifest)


def test_duplicate_entries_in_pinned_checksum_list_are_rejected(package):
    path = package.model / "SHA256SUMS"
    data = path.read_bytes()
    data += data.splitlines(keepends=True)[0]
    path.write_bytes(data)
    package.manifest["sha256"]["SHA256SUMS"] = sha(data)
    with pytest.raises(ValueError, match="Duplicate published filename"):
        download.checksums(package.model, package.manifest)


@pytest.mark.parametrize("inside", [".", "metadata"])
def test_metadata_output_must_be_outside_checkpoint(package, inside):
    with pytest.raises(ValueError, match="outside the model directory"):
        metadata.prepare(package.model, package.model / inside, package.manifest)


def test_output_containment_is_checked_after_resolving_symlinks(package):
    alias = package.model.parent / "alias"
    alias.symlink_to(package.model, target_is_directory=True)
    with pytest.raises(ValueError, match="outside the model directory"):
        metadata.prepare(package.model, alias / "metadata", package.manifest)


def test_existing_output_is_not_overwritten(package):
    package.output.mkdir()
    sentinel = package.output / "keep.txt"
    sentinel.write_text("keep")
    with pytest.raises(ValueError, match="Output already exists"):
        metadata.prepare(package.model, package.output, package.manifest)
    assert sentinel.read_text() == "keep"


def test_sidecar_cannot_reference_files_outside_checkpoint(package):
    package.sidecar["files"][0]["path"] = "../../external.safetensors"
    write_json(package.overlay / metadata.SIDECAR, package.sidecar)
    pin_fixture(package.model, package.manifest)
    with pytest.raises(ValueError, match="sidecar path outside checkpoint"):
        metadata.prepare(package.model, package.output, package.manifest)
    assert not package.output.exists()


def test_sidecar_header_hash_mismatch_is_rejected(package):
    package.sidecar["files"][0]["header_sha256"] = "0" * 64
    write_json(package.overlay / metadata.SIDECAR, package.sidecar)
    pin_fixture(package.model, package.manifest)
    with pytest.raises(ValueError, match="Pinned safetensors header mismatch"):
        metadata.prepare(package.model, package.output, package.manifest)
    assert not package.output.exists()


@pytest.mark.parametrize("field", ["source_config_sha256", "source_index_sha256"])
def test_sidecar_must_bind_to_the_downloaded_config_and_index(package, field):
    package.sidecar[field] = "f" * 64
    write_json(package.overlay / metadata.SIDECAR, package.sidecar)
    pin_fixture(package.model, package.manifest)
    with pytest.raises(ValueError, match="mismatched published exact-metadata proof"):
        metadata.prepare(package.model, package.output, package.manifest)
    assert not package.output.exists()


@pytest.mark.parametrize("field", ["attention_output_quantization", "preserve_mtp"])
def test_required_1017_model_features_cannot_be_dropped(package, field):
    path = package.model / "config.json"
    config = json.loads(path.read_text())
    config["omlx_deepseek_v41"][field] = False
    write_json(path, config)
    pin_fixture(package.model, package.manifest)
    with pytest.raises(ValueError, match="required 1017 configuration"):
        metadata.prepare(package.model, package.output, package.manifest)
    assert not package.output.exists()


@pytest.mark.parametrize("raw", [b"short", b"\0" * 8, struct.pack("<Q", 2**32)])
def test_header_reader_rejects_truncation_and_unbounded_lengths(package, raw):
    path = package.model / "bad-header.safetensors"
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="(Truncated|Invalid) safetensors header"):
        metadata.header_identity(path, package.header_pin)
