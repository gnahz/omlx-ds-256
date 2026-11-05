"""CPU regression tests for 0.7.3 CED replay and custom cache/I/O ownership."""
import hashlib
import json
import struct

import mlx.core as mx
import numpy as np
import pytest

from omlx.patches.deepseek_v41 import prefix_cache_store as store, storage
from omlx.patches.deepseek_v41.prefix_cache import ExactPrefixCache, cache_arrays
from test_deepseek_v41_prefix_cache import sample
from test_deepseek_v41_prefix_cache_store import FINGERPRINT, current, flush, manager
from test_deepseek_v41_engram_io import table


@pytest.fixture(autouse=True)
def cpu_only():
    before = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(before)


def pending_record(result):
    cache = sample()
    cache[0].set_pending((
        mx.arange(48).reshape(1, 3, 2, 8).astype(mx.bfloat16),
        mx.arange(6).reshape(1, 3, 2).astype(mx.float32),
        mx.arange(9).reshape(1, 3, 3).astype(mx.int32),
        mx.zeros((1, 0, 0), mx.int32),
    ))
    tokens = list(range(9000))
    _, _, plan = result.prepare(tokens)
    result.record(plan, cache, 0, 8192)
    return tokens, cache


def assert_identical(left, right):
    assert left[0].meta_state == right[0].meta_state == ("deepseek_v41", "4", "2")
    assert left[0].has_pending() and right[0].has_pending()
    aa, bb = cache_arrays(left), cache_arrays(right)
    assert len(aa) == len(bb)
    for a, b in zip(aa, bb):
        assert a.shape == b.shape and a.dtype == b.dtype
        np.testing.assert_array_equal(np.asarray(a.view(mx.uint8)), np.asarray(b.view(mx.uint8)))


def test_pending_decoder_replay_restart_and_request_copy(tmp_path):
    first, second = manager(tmp_path), manager(tmp_path)
    try:
        tokens, original = pending_record(first)
        before, hit, _ = first.prepare(tokens)
        assert hit == 8192
        assert_identical(original, before)
        flush(first)
        assert second.restore_persisted() == 1
        after, hit, _ = second.prepare(tokens)
        assert hit == 8192
        assert_identical(before, after)
        after[0].set_pending(None)
        untouched, _, _ = second.prepare(tokens)
        assert_identical(before, untouched)
    finally:
        first.close_persistence()
        second.close_persistence()


def test_pending_decoder_replay_survives_l2_eviction_and_reopen(tmp_path):
    first = ExactPrefixCache(2**20)
    first.configure_l2(tmp_path, FINGERPRINT, 4 * 2**20)
    try:
        tokens, before = pending_record(first)
        assert first.flush_l2(10)
        first.clear()
        after, hit, _ = first.prepare(tokens)
        assert hit == 8192
        assert_identical(before, after)
    finally:
        first.close_l2(timeout_seconds=10)
    second = ExactPrefixCache(2**20)
    second.configure_l2(tmp_path, FINGERPRINT, 4 * 2**20)
    try:
        after, hit, _ = second.prepare(tokens)
        assert hit == 8192
        assert_identical(before, after)
    finally:
        second.close_l2(timeout_seconds=10)


def test_pending_decoder_replay_bad_metadata_rejected_before_load(tmp_path, monkeypatch):
    first = manager(tmp_path)
    pending_record(first)
    flush(first)
    manifest, descriptor, path = current(tmp_path)
    data = path.read_bytes()
    size = struct.unpack("<Q", data[:8])[0]
    header = json.loads(data[8:8 + size])
    # A valid checksum cannot disguise a v3 layer that carries v4 replay slots.
    descriptor["layout"]["layers"][0]["meta_state"][1] = "3"
    header["__metadata__"]["layout"] = store._json_bytes(descriptor["layout"]).decode()
    encoded = store._json_bytes(header)
    encoded += b" " * (-len(encoded) % 8)
    payload = struct.pack("<Q", len(encoded)) + encoded + data[8 + size:]
    path.write_bytes(payload)
    descriptor["file_bytes"] = len(payload)
    descriptor["sha256"] = hashlib.sha256(payload).hexdigest()
    tmp_path.joinpath("current.json").write_text(json.dumps(manifest))
    second = manager(tmp_path)
    try:
        with monkeypatch.context() as patch:
            patch.setattr(mx, "load", lambda *a, **kw: pytest.fail("invalid checkpoint allocated"))
            assert second.restore_persisted() == 0
        assert second.snapshot_stats()["persistence"]["restore_skipped"] == 1
    finally:
        first.close_persistence()
        second.close_persistence()


def test_engram_carry_activates_into_positional_queue_and_reuses_shortened_chunk(table):
    embed = storage.DiskEngramEmbedding(table, "weight", "scale")
    prefetch = storage.EngramPrefetch()
    ids = np.random.default_rng(73).integers(0, 5000, (1, 40, 24))
    ids[:, 20:30] = ids[:, :10]  # tuple dedup must not reorder queued positions
    expected = np.asarray(embed(mx.array(ids)).astype(mx.float32))
    try:
        assert prefetch.submit_carry(embed, ids)
        assert prefetch.has_pending() and not embed._queue.items
        assert prefetch.activate() and not prefetch.has_pending()
        for a, b in ((0, 13), (13, 40)):
            actual = np.asarray(embed(mx.array(ids[:, a:b])).astype(mx.float32))
            np.testing.assert_array_equal(actual, expected[:, a:b])
        assert embed._queue.stats["claimed"] == ids.size
        assert embed._queue.stats["read_at_claim"] == 0
        assert not embed._queue.items and not prefetch.activate()
    finally:
        prefetch.close()
        embed.close()


def test_engram_carry_budget_and_drain(table, monkeypatch):
    embed = storage.DiskEngramEmbedding(table, "weight", "scale")
    prefetch = storage.EngramPrefetch()
    ids = np.arange(24, dtype=np.int64).reshape(1, 1, 24)
    try:
        monkeypatch.setattr(storage, "LOOKAHEAD_MAX_BYTES", embed.selected_bytes(ids.size))
        assert not prefetch.submit_carry(embed, ids)  # IDs count in carry budget too
        monkeypatch.setattr(storage, "LOOKAHEAD_MAX_BYTES", 2**20)
        assert prefetch.submit_carry(embed, ids)
        prefetch.drain()
        assert not prefetch.has_pending() and not embed._queue.items
    finally:
        prefetch.close()
        embed.close()
