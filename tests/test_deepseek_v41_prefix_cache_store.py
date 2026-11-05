"""Restart cache persistence with real MLX arrays and filesystem transactions."""

import hashlib
import json
from pathlib import Path
import struct
import time

import mlx.core as mx
import numpy as np
import pytest
from safetensors import safe_open

from omlx.patches.deepseek_v41.prefix_cache import ExactPrefixCache, cache_arrays
from omlx.patches.deepseek_v41 import prefix_cache_store as store
from test_deepseek_v41_prefix_cache import (
    sample, image_tokens, image_identity, record_image, DOMAIN, IMAGE_A, IMAGE_B, IMAGE_C,
)

FINGERPRINT = {"model": "test-model", "tokenizer": "test-tokens", "program": "CED8192+DSpark"}


def manager(directory, max_bytes=2**20, fingerprint=None):
    result = ExactPrefixCache(max_bytes)
    result.configure_persistence(directory, fingerprint or FINGERPRINT, interval_seconds=300)
    return result


def add(result, end=8192):
    tokens = list(range(end + 17))
    cache = sample(end)
    cache[0].left_padding = mx.array([0], mx.int32)
    cache[0].lengths = mx.array([end], mx.int32)
    # Include raw bfloat16 bit patterns including signed zero and NaN payloads.
    cache[0].cache[1] = mx.arange(24).astype(mx.uint16).reshape(1, 8, 3).view(mx.bfloat16)
    _, _, plan = result.prepare(tokens)
    # Build every complete checkpoint as a real prefill would.
    for start in range(plan.next_position, end, 8192):
        cache[0].cache[0] = mx.array([start + 8192])
        cache[0]._omlx_mtp_prime_ctx.expected_target_offset = start + 8192
        cache[0]._omlx_mtp_prime_ctx.caches[0].offset = start + 8192
        result.record(plan, cache, start, 8192)
    return tokens


def flush(result):
    deadline = time.monotonic() + 10
    while result.persistence_step(force=True):
        assert time.monotonic() < deadline, "save did not finish"
        time.sleep(0.001)
    return result.snapshot_stats()["persistence"]


def current(directory):
    manifest = json.loads((directory / "current.json").read_text())
    descriptor = manifest["entries"][-1]
    path = directory / manifest["generation"] / (descriptor["key"] + ".safetensors")
    return manifest, descriptor, path


def test_disk_roundtrip_all_state_bits_lru_and_no_request_reads(tmp_path, monkeypatch):
    first = manager(tmp_path)
    tokens = add(first, 16384)
    before = first.prepare(tokens)[0]
    stats = flush(first)
    assert not stats["dirty"] and stats["saves"] == 1
    assert stats["saved_generation"]
    _, _, file = current(tmp_path)
    with safe_open(file, framework="numpy") as tensors:
        assert "tokens" in tensors.keys() and "prime.0" in tensors.keys()
    second = manager(tmp_path)
    assert second.restore_persisted() == 2
    assert list(second.entries) == list(first.entries)
    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("disk read")))
        after, cached, _ = second.prepare(tokens)
    assert cached == 16384
    for left, right in zip(cache_arrays(before), cache_arrays(after)):
        assert left.dtype == right.dtype and left.shape == right.shape
        assert np.array_equal(np.asarray(left.view(mx.uint8)), np.asarray(right.view(mx.uint8)))
    a, b = before[0]._omlx_mtp_prime_ctx, after[0]._omlx_mtp_prime_ctx
    assert a.expected_target_offset == b.expected_target_offset == 16384
    assert a.caches[0].max_size == b.caches[0].max_size
    assert a.caches[0].offset == b.caches[0].offset
    first.close_persistence()
    second.close_persistence()


def test_interval_and_incremental_one_checkpoint_staging(tmp_path):
    result = manager(tmp_path)
    add(result, 16384)
    assert not result.persistence_step() and not tmp_path.joinpath("current.json").exists()
    assert result.persistence_step(force=True)
    assert result.persistence_pending
    assert not tmp_path.joinpath("current.json").exists()
    assert not flush(result)["dirty"]
    # No changes means no unnecessary new generation on normal stop.
    assert not result.persistence_step(force=True)
    assert result.snapshot_stats()["persistence"]["saves"] == 1
    result.close_persistence()


def test_fingerprint_mismatch_and_memory_budget_skip_before_mlx_load(tmp_path, monkeypatch):
    result = manager(tmp_path)
    add(result)
    flush(result)
    wrong = manager(tmp_path, fingerprint={"model": "different"})
    assert wrong.restore_persisted() == 0
    limited = manager(tmp_path)
    limited.ensure_headroom = lambda size: False
    with monkeypatch.context() as patch:
        patch.setattr(mx, "load", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("allocated")))
        assert limited.restore_persisted() == 0
    assert limited.skipped_memory == 1
    for item in (result, wrong, limited):
        item.close_persistence()


def test_restart_restores_most_recent_that_fit(tmp_path):
    result = manager(tmp_path)
    add(result, 16384)
    largest = list(result.entries.values())[-1].size
    flush(result)
    loaded = manager(tmp_path, max_bytes=largest)
    assert loaded.restore_persisted() == 1
    assert loaded.prepare(list(range(17000)))[1] == 16384
    result.close_persistence()
    loaded.close_persistence()


def test_failed_write_preserves_last_atomic_generation(tmp_path, monkeypatch):
    result = manager(tmp_path)
    tokens = add(result)
    flush(result)
    original = tmp_path.joinpath("current.json").read_bytes()
    add(result, 16384)
    with monkeypatch.context() as patch:
        patch.setattr(store, "_write_checkpoint", lambda *a: (_ for _ in ()).throw(OSError("disk full")))
        stats = flush(result)
    assert stats["dirty"] and stats["failures"] == 1 and "disk full" in stats["last_error"]
    assert tmp_path.joinpath("current.json").read_bytes() == original
    loaded = manager(tmp_path)
    assert loaded.restore_persisted() == 1 and loaded.prepare(tokens)[1] == 8192
    result.close_persistence()
    loaded.close_persistence()


def test_corruption_rejected_before_loading_and_previous_generation_survives(tmp_path, monkeypatch):
    result = manager(tmp_path)
    add(result)
    flush(result)
    add(result, 16384)
    flush(result)
    manifest, descriptor, path = current(tmp_path)
    data = bytearray(path.read_bytes())
    data[-1] ^= 1
    path.write_bytes(data)
    loaded = manager(tmp_path)
    # The intact 8K entry remains usable; corrupted 16K is skipped.
    assert loaded.restore_persisted() == 1
    assert loaded.prepare(list(range(17000)))[1] == 8192
    assert loaded.snapshot_stats()["persistence"]["restore_skipped"] == 1
    assert tmp_path.joinpath("previous.json").exists()
    result.close_persistence()
    loaded.close_persistence()


def test_invalid_shape_rejected_before_mlx_allocates_even_with_correct_checksum(tmp_path, monkeypatch):
    result = manager(tmp_path)
    add(result)
    flush(result)
    manifest, descriptor, path = current(tmp_path)
    data = path.read_bytes()
    length = struct.unpack("<Q", data[:8])[0]
    header = json.loads(data[8:8 + length])
    header["layer.0.1"]["shape"] = [1, 999999999, 3]
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    payload = struct.pack("<Q", len(encoded)) + encoded + data[8 + length:]
    path.write_bytes(payload)
    descriptor["sha256"] = hashlib.sha256(payload).hexdigest()
    descriptor["file_bytes"] = len(payload)
    tmp_path.joinpath("current.json").write_text(json.dumps(manifest))
    loaded = manager(tmp_path)
    with monkeypatch.context() as patch:
        patch.setattr(mx, "load", lambda *a, **kw: pytest.fail("malformed checkpoint allocated"))
        assert loaded.restore_persisted() == 0
    assert loaded.snapshot_stats()["persistence"]["restore_skipped"] == 1
    result.close_persistence()
    loaded.close_persistence()


def test_memory_save_deferral_retains_committed_data_and_dirty_state(tmp_path):
    result = manager(tmp_path)
    add(result)
    flush(result)
    previous = tmp_path.joinpath("current.json").read_bytes()
    add(result, 16384)
    result.ensure_headroom = lambda size: False
    stats = flush(result)
    assert stats["dirty"] and "insufficient memory" in stats["last_error"]
    assert tmp_path.joinpath("current.json").read_bytes() == previous
    result.close_persistence()


def test_idle_save_reads_no_payload_and_worker_never_stages_mlx(tmp_path, monkeypatch):
    import threading

    initial = manager(tmp_path)
    add(initial)
    flush(initial)
    restored = manager(tmp_path)
    assert restored.restore_persisted() == 1
    reads = restored.snapshot_stats()["persistence"]["disk_reads"]
    assert reads == 2
    add(restored, 16384)
    main_thread = threading.get_ident()
    stage, write = store._stage, store._write_checkpoint

    def checked_stage(*args):
        assert threading.get_ident() == main_thread
        return stage(*args)

    def checked_write(*args):
        assert threading.get_ident() != main_thread
        return write(*args)

    with monkeypatch.context() as patch:
        patch.setattr(store, "_stage", checked_stage)
        patch.setattr(store, "_write_checkpoint", checked_write)
        patch.setattr(store, "_validate_file", lambda *a: pytest.fail("save reread a payload"))
        patch.setattr(mx, "load", lambda *a: pytest.fail("save loaded MLX state"))
        assert not flush(restored)["dirty"]
    assert restored.snapshot_stats()["persistence"]["disk_reads"] == reads
    for item in (initial, restored):
        item.close_persistence()


def test_atomic_manifest_failure_keeps_previous_current_usable(tmp_path, monkeypatch):
    result = manager(tmp_path)
    add(result)
    flush(result)
    previous = tmp_path.joinpath("current.json").read_bytes()
    add(result, 16384)
    atomic = store._atomic_json

    def fail_commit(path, value):
        if path.name == "current.json":
            raise OSError("interrupted manifest commit")
        atomic(path, value)

    with monkeypatch.context() as patch:
        patch.setattr(store, "_atomic_json", fail_commit)
        stats = flush(result)
    assert stats["dirty"] and "interrupted manifest" in stats["last_error"]
    assert tmp_path.joinpath("current.json").read_bytes() == previous
    restarted = manager(tmp_path)
    assert restarted.restore_persisted() == 1
    for item in (result, restarted):
        item.close_persistence()


def test_malformed_layout_cannot_change_draft_state_without_checksum(tmp_path):
    result = manager(tmp_path)
    add(result)
    flush(result)
    manifest, descriptor, _ = current(tmp_path)
    descriptor["layout"]["prime"]["stages"][0]["max_size"] = 999
    tmp_path.joinpath("current.json").write_text(json.dumps(manifest))
    restarted = manager(tmp_path)
    assert restarted.restore_persisted() == 0
    assert "layout metadata mismatch" in restarted.snapshot_stats()["persistence"]["last_error"]
    for item in (result, restarted):
        item.close_persistence()


def test_generation_replacement_disk_peak_has_at_most_two_datasets(tmp_path):
    result = manager(tmp_path)
    for end in (8192, 16384, 24576):
        add(result, end)
        deadline = time.monotonic() + 10
        while result.persistence_step(force=True):
            assert len(list(tmp_path.glob("generation-*"))) <= 2
            assert time.monotonic() < deadline
            time.sleep(0.001)
    assert len(list(tmp_path.glob("generation-*"))) == 2
    result.close_persistence()


def test_repeated_failed_initial_saves_reclaim_uncommitted_generations(tmp_path, monkeypatch):
    result = manager(tmp_path)
    add(result)
    with monkeypatch.context() as patch:
        patch.setattr(store, "_write_checkpoint", lambda *a: (_ for _ in ()).throw(OSError("disk full")))
        for attempt in range(4):
            stats = flush(result)
            assert stats["dirty"] and stats["failures"] == attempt + 1
            assert len(list(tmp_path.glob("generation-*"))) == 1
            assert not tmp_path.joinpath("current.json").exists()
    assert not flush(result)["dirty"]
    assert len(list(tmp_path.glob("generation-*"))) == 1
    restarted = manager(tmp_path)
    assert restarted.restore_persisted() == 1
    for item in (result, restarted):
        item.close_persistence()


@pytest.mark.parametrize("skip_reason", ["headroom", "capacity"])
def test_all_skipped_restore_preserves_durable_generation_on_empty_save(tmp_path, skip_reason):
    initial = manager(tmp_path)
    tokens = add(initial)
    flush(initial)
    saved, _, _ = current(tmp_path)
    initial.close_persistence()
    restarted = manager(tmp_path, max_bytes=1 if skip_reason == "capacity" else 2**20)
    if skip_reason == "headroom":
        restarted.ensure_headroom = lambda size: False
    assert restarted.restore_persisted() == 0
    assert restarted._persistence._committed_manifest["generation"] == saved["generation"]
    assert not restarted.snapshot_stats()["persistence"]["dirty"]
    # A later administrative clear/new empty save must keep the prior durable
    # generation as the recovery backup, even though it never fit in memory.
    restarted.clear()
    assert not flush(restarted)["dirty"]
    backup = json.loads(tmp_path.joinpath("previous.json").read_text())
    assert backup["generation"] == saved["generation"]
    assert backup["entries"] == saved["entries"]
    assert tmp_path.joinpath(saved["generation"]).is_dir()
    restarted.close_persistence()
    # Simulate an interrupted/corrupt newest manifest: the retained backup
    # still restores the original cache when sufficient memory is available.
    tmp_path.joinpath("current.json").write_text("interrupted")
    recovered = manager(tmp_path)
    assert recovered.restore_persisted() == 1
    assert recovered.prepare(tokens)[1] == 8192
    recovered.close_persistence()


def test_multimodal_restart_restores_exact_identity_and_stable_early_prefix(tmp_path):
    first = manager(tmp_path)
    tokens = image_tokens()
    ranges = [(0, (DOMAIN,)), (100, (IMAGE_A,)), (9000, (IMAGE_B,))]
    record_image(first, tokens, ranges)
    flush(first)
    _, descriptor, path = current(tmp_path)
    with safe_open(path, framework="numpy") as tensors:
        assert json.loads(tensors.metadata()["identity"]) == descriptor["identity"]
    first.close_persistence()
    second = manager(tmp_path)
    try:
        assert second.restore_persisted() == 2
        assert second.prepare(tokens, **image_identity(ranges))[1] == 16384
        changed = [(0, (DOMAIN,)), (100, (IMAGE_C,)), (9000, (IMAGE_B,))]
        assert second.prepare(tokens, **image_identity(changed))[1] == 0
        changed = [(0, (DOMAIN,)), (100, (IMAGE_A,)), (9000, (IMAGE_C,))]
        assert second.prepare(tokens, **image_identity(changed))[1] == 8192
        assert second.prepare(tokens) == (None, 0, None)
        appended = tokens + list(range(17000, 26000))
        appended[20000] = 129264
        assert second.prepare(appended, **image_identity(ranges + [(20000, (IMAGE_C,))]))[1] == 16384
    finally:
        second.close_persistence()


@pytest.mark.parametrize("corruption", ["missing", "different", "malformed", "legacy"])
def test_image_identity_manifest_tampering_is_rejected_before_mlx_load(tmp_path, monkeypatch, corruption):
    first = manager(tmp_path)
    tokens = image_tokens(9000, (100,))
    ranges = [(0, (DOMAIN,)), (100, (IMAGE_A,))]
    record_image(first, tokens, ranges)
    flush(first)
    manifest, descriptor, _ = current(tmp_path)
    first.close_persistence()
    if corruption == "missing":
        descriptor.pop("identity")
    elif corruption == "different":
        descriptor["identity"][-1][1] = [IMAGE_B]
    elif corruption == "malformed":
        descriptor["identity"] = [[-1, [IMAGE_A]]]
    else:
        manifest["format"] = "ds41-exact-prefix-v1"
    tmp_path.joinpath("current.json").write_text(json.dumps(manifest))
    restarted = manager(tmp_path)
    try:
        with monkeypatch.context() as patch:
            patch.setattr(mx, "load", lambda *a, **kw: pytest.fail("invalid identity allocated"))
            assert restarted.restore_persisted() == 0
        assert restarted.snapshot_stats()["persistence"]["restore_skipped"] >= 1
    finally:
        restarted.close_persistence()


def test_changed_proof_and_updated_payload_checksum_still_fail_identity_digest(tmp_path):
    first = manager(tmp_path)
    tokens = image_tokens(9000, (100,))
    ranges = [(0, (DOMAIN,)), (100, (IMAGE_A,))]
    record_image(first, tokens, ranges)
    flush(first)
    manifest, descriptor, path = current(tmp_path)
    first.close_persistence()
    data = path.read_bytes()
    length = struct.unpack("<Q", data[:8])[0]
    header = json.loads(data[8:8 + length])
    descriptor["identity"][-1][1] = [IMAGE_B]
    header["__metadata__"]["identity"] = json.dumps(descriptor["identity"], separators=(",", ":"))
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    payload = struct.pack("<Q", len(encoded)) + encoded + data[8 + length:]
    path.write_bytes(payload)
    descriptor["sha256"] = hashlib.sha256(payload).hexdigest()
    descriptor["file_bytes"] = len(payload)
    tmp_path.joinpath("current.json").write_text(json.dumps(manifest))
    restarted = manager(tmp_path)
    try:
        assert restarted.restore_persisted() == 0
        assert "token digest mismatch" in restarted.snapshot_stats()["persistence"]["last_error"]
    finally:
        restarted.close_persistence()
