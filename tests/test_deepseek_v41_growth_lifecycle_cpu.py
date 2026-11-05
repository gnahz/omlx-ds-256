"""CPU-only correctness of growth across rollback, batching and checkpoint I/O.

These tests use the real storage/rollback/persistence implementations with small
arrays. They never load a model or invoke a target backbone/Metal kernel. The
checkpoint seed is an already-owned live prefix, so the test exercises actual
stage/checksum/load/restore without calling the GPU-only compact-copy kernel.
"""
from types import SimpleNamespace
import hashlib
import struct
import time

import mlx.core as mx
import numpy as np
import pytest

from omlx.patches.deepseek_v41 import growth
from omlx.patches.deepseek_v41.cache import DeepseekV41Cache
from omlx.patches.deepseek_v41.mtp import DSparkMixin
from omlx.patches.deepseek_v41.prefix_cache import (
    CHUNK, Entry, ExactPrefixCache, cache_arrays, clone_cache,
)
from omlx.patches.mlx_lm_mtp.deepseek_v4_dspark import (
    DSparkContextCache, _DSparkPrimeContext,
)


@pytest.fixture(autouse=True)
def cpu_growth(monkeypatch):
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    monkeypatch.setattr(growth, "ENABLED", True)
    monkeypatch.setattr(growth, "PINGPONG", True)

    def reject_metal(*args, **kwargs):
        raise AssertionError("CPU lifecycle tests must not construct Metal kernels")

    monkeypatch.setattr(mx.fast, "metal_kernel", reject_metal)
    yield
    mx.synchronize()
    mx.set_default_device(previous)


def rows(count, width, seed):
    return np.random.default_rng(seed).integers(0, 256, (1, count, width), dtype=np.uint8)


def raw(array):
    return None if array is None else np.asarray(array.view(mx.uint8)).copy()


def assert_bits(actual, expected):
    if actual is None or expected is None:
        assert actual is expected
    else:
        assert actual.shape == expected.shape and actual.dtype == expected.dtype
        np.testing.assert_array_equal(raw(actual), raw(expected))


def seed_slot(cache, slot, values):
    cache[slot] = mx.zeros((1, 0, values.shape[-1]), mx.uint8)
    # Build a real idle/current registry rather than only testing first append.
    split = [max(0, values.shape[1] - 64), max(0, values.shape[1] - 32), values.shape[1]]
    start = 0
    for end in split:
        if end > start:
            cache[slot] = growth.append(cache, slot, cache[slot], mx.array(values[:, start:end]), start)
            start = end


def cache_at(tokens, ratio, seed=0):
    cache = DeepseekV41Cache(ratio)
    cache[0] = mx.array([tokens], mx.int32)
    cache[1] = mx.array(rows(min(tokens, 8), 528, seed + 1))
    seed_slot(cache, 2, rows(tokens // ratio, 288, seed + 2))
    seed_slot(cache, 3, rows(tokens // ratio, 68, seed + 3))
    cache[4] = mx.array(rows(tokens % ratio, 512, seed + 4)).astype(mx.float32)
    cache[5] = mx.array(rows(tokens % ratio, 512, seed + 5)).astype(mx.float32)
    cache[6] = mx.array([[7, 8, 9]], mx.int64)
    return cache


class History:
    def __call__(self, ids, history):
        joined = np.concatenate([np.asarray(history), np.asarray(ids)], axis=1)
        return None, joined[:, -3:]


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("ratio", [1, 2, 4])
@pytest.mark.parametrize("before", [17, 18])
@pytest.mark.parametrize("accepted", [0, 1, 3])
def test_real_mtp_rollback_and_following_append_match_committed_prefix(
    monkeypatch, enabled, ratio, before, accepted,
):
    """Exercise actual rollback at compression boundaries, including full accept."""
    monkeypatch.setattr(growth, "ENABLED", enabled)
    cache = cache_at(before, ratio, seed=31)
    cache.left_padding, cache.lengths = mx.array([12], mx.int32), mx.array([40], mx.int32)
    snapshot = (list(cache.cache), cache.left_padding, cache.lengths)
    old_bits = [raw(value) for value in snapshot[0]]
    inputs = mx.array([[21, 22, 23, 24]], mx.int32)
    count = accepted + 1
    end = before + count
    verified_end = before + inputs.shape[1]
    new_count = verified_end // ratio - before // ratio
    values = {slot: rows(new_count, width, slot + 99) for slot, width in ((2, 288), (3, 68))}
    for slot in (2, 3):
        cache[slot] = growth.append(cache, slot, cache[slot], mx.array(values[slot]), before // ratio)
    window = mx.concatenate([snapshot[0][1], mx.array(rows(4, 528, 90))], axis=1)
    compressor = tuple(mx.concatenate([snapshot[0][slot],
        mx.array(rows(4, 512, slot + 91)).astype(mx.float32)], axis=1) for slot in (4, 5))
    projected_end = before % ratio + 4
    cutoff = projected_end // ratio * ratio
    cache[0] = mx.array([verified_end], mx.int32)
    cache[1] = window[:, -8:]
    if ratio > 1:
        cache[4], cache[5] = (value[:, cutoff:] for value in compressor)
    cache[6] = mx.array([[22, 23, 24]], mx.int64)
    cache.left_padding, cache.lengths = snapshot[1] - 4, snapshot[2] - 4
    cache._mtp_draft_stash = (inputs, [snapshot], before,
        [{"window": window, "compressor": compressor}])
    owner = SimpleNamespace(_config=SimpleNamespace(window_size=8), _hasher=History())

    assert DSparkMixin.mtp_partial_rollback(owner, [cache], accepted, 3)
    assert cache._mtp_draft_stash is None
    assert cache.size() == end
    np.testing.assert_array_equal(np.asarray(cache.left_padding), [12 - count])
    np.testing.assert_array_equal(np.asarray(cache.lengths), [40 - count])
    committed_rows = end // ratio - before // ratio
    for slot in (2, 3):
        expected = np.concatenate([np.asarray(snapshot[0][slot]), values[slot][:, :committed_rows]], axis=1)
        np.testing.assert_array_equal(np.asarray(cache[slot]), expected)
    expected_window_end = min(before, 8) + count
    assert_bits(cache[1], window[:, max(0, expected_window_end - 8):expected_window_end])
    if ratio > 1:
        projected_end = before % ratio + count
        cutoff = projected_end // ratio * ratio
        for slot, value in zip((4, 5), compressor):
            assert_bits(cache[slot], value[:, cutoff:projected_end])
    np.testing.assert_array_equal(np.asarray(cache[6]), np.concatenate(
        [np.asarray(snapshot[0][6]), np.asarray(inputs)[:, :count]], axis=1)[:, -3:])

    # A new verify/decode append must not revive rejected rows from the idle bank.
    for slot, width in ((2, 288), (3, 68)):
        committed = np.asarray(cache[slot]).copy()
        addition = rows(2, width, slot + 111)
        cache[slot] = growth.append(cache, slot, cache[slot], mx.array(addition), end // ratio)
        np.testing.assert_array_equal(np.asarray(cache[slot]), np.concatenate([committed, addition], axis=1))
    # Retained causal snapshots keep their bytes despite registry reuse/COW.
    for value, original in zip(snapshot[0], old_bits):
        np.testing.assert_array_equal(raw(value), original)


def test_ragged_batch_filter_reorder_and_interleaved_growth_keep_requests_separate():
    originals = {name: cache_at(tokens, 2, seed) for name, tokens, seed in
                 (("a", 14, 1), ("b", 22, 11), ("c", 18, 21))}
    expected = {name: {slot: np.asarray(cache[slot]).copy() for slot in (2, 3)}
                for name, cache in originals.items()}
    tokens = {name: cache.size() for name, cache in originals.items()}
    order = ["a", "b"]
    batch = DeepseekV41Cache.merge([originals[name] for name in order])
    for step in range(18):
        extracted = []
        for row, name in enumerate(order):
            cache = batch.extract(row, offset=tokens[name])
            add = 1 + (step + row) % 5
            start = tokens[name] // 2
            keep = start + add - (1 if step % 3 == 0 else 0)
            for slot, width in ((2, 288), (3, 68)):
                addition = rows(add, width, 1000 + step * 10 + row + slot)
                cache[slot] = growth.append(cache, slot, cache[slot][:, :start], mx.array(addition), start)
                growth.truncate(cache, slot, keep)
                expected[name][slot] = np.concatenate([expected[name][slot], addition], axis=1)[:, :keep]
                np.testing.assert_array_equal(np.asarray(cache[slot]), expected[name][slot])
            tokens[name] = keep * 2
            cache[0] = mx.array([tokens[name]], mx.int32)
            extracted.append(cache)
        batch = DeepseekV41Cache.merge(extracted)
        if step == 3:
            batch.filter([1, 0])
            order.reverse()
            batch.extend(originals["c"])
            order.append("c")
        elif step == 10:
            batch.filter([2, 0])
            order = [order[2], order[0]]
        elif step == 14:
            batch.filter([1])
            order = [order[1]]
        for row, name in enumerate(order):
            restored = batch.extract(row, offset=tokens[name])
            for slot in (2, 3):
                np.testing.assert_array_equal(np.asarray(restored[slot]), expected[name][slot])
        if batch.batch_size == 1:
            assert batch.extract(0)._ds41_buffers is batch._ds41_buffers


def add_prime_context(cache, offset):
    stages = []
    for seed in (41, 42, 43):
        stage = DSparkContextCache(128)
        stage.offset = offset
        stage.keys = mx.array(rows(8, 16, seed).reshape(1, 1, 8, 16))
        stages.append(stage)
    cache._omlx_mtp_prime_ctx = _DSparkPrimeContext(stages, offset)


def seeded_manager(directory, ratio):
    live = cache_at(CHUNK, ratio, seed=51)
    live.left_padding = mx.array([0], mx.int32)
    live.lengths = mx.array([CHUNK], mx.int32)
    add_prime_context(live, CHUNK)
    live._mtp_verify_state = {"private": "must not serialize"}
    live._mtp_draft_stash = object()
    snapshot = clone_cache([live])
    assert not hasattr(snapshot[0], "_ds41_buffers")
    assert not hasattr(snapshot[0], "_mtp_verify_state")
    assert not hasattr(snapshot[0], "_mtp_draft_stash")
    tokens = list(range(CHUNK + 17))
    packed = struct.pack(f"<{CHUNK}I", *tokens[:CHUNK])
    key = hashlib.sha256(b"ds41-ced-8192-v1" + packed).digest()
    entry = Entry(packed, snapshot, sum(value.nbytes for value in cache_arrays(snapshot)) + len(packed))
    manager = ExactPrefixCache(16 << 20)
    manager.entries[key], manager.bytes = entry, entry.size
    manager._revision += 1
    fingerprint = {"runtime": "growth-lifecycle-cpu-v1", "storage_policy": {"growth": True, "pingpong": True}}
    manager.configure_persistence(directory, fingerprint, interval_seconds=300)
    return manager, tokens, fingerprint


def assert_independent_readers(manager, tokens):
    first, reused, _ = manager.prepare(tokens)
    second, other_reused, _ = manager.prepare(tokens)
    assert reused == other_reused == CHUNK
    snapshot = next(iter(manager.entries.values())).cache
    before = [raw(value) for value in cache_arrays(second)]
    for cache in (first, second):
        assert not hasattr(cache[0], "_ds41_buffers")
        assert not hasattr(cache[0], "_mtp_verify_state")
        assert cache[0]._omlx_mtp_prime_ctx is not snapshot[0]._omlx_mtp_prime_ctx
    for slot, width in ((2, 288), (3, 68)):
        a = first[0]
        original = np.asarray(a[slot]).copy()
        start = original.shape[1]
        a[slot] = growth.append(a, slot, a[slot], mx.array(rows(3, width, slot + 81)), start)
        growth.truncate(a, slot, start + 1)
        a[slot] = growth.append(a, slot, a[slot], mx.array(rows(2, width, slot + 82)), start + 1)
        assert a[slot].shape[1] == start + 3
        np.testing.assert_array_equal(np.asarray(a[slot])[:, :start], original)
    first[0]._omlx_mtp_prime_ctx.caches[0].keys[:] = 0
    first[0]._omlx_mtp_prime_ctx.caches[0].offset += 1
    for value, original in zip(cache_arrays(second), before):
        np.testing.assert_array_equal(raw(value), original)
    for value, original in zip(cache_arrays(snapshot), before):
        np.testing.assert_array_equal(raw(value), original)
    assert second[0]._omlx_mtp_prime_ctx.caches[0].offset == CHUNK


@pytest.mark.parametrize("ratio", [1, 2])
def test_exact_prefix_clone_and_disk_restore_reseed_growth_without_shared_mutable_state(tmp_path, ratio, monkeypatch):
    manager, tokens, fingerprint = seeded_manager(tmp_path, ratio)
    loaded, mismatched = ExactPrefixCache(16 << 20), ExactPrefixCache(16 << 20)
    try:
        assert_independent_readers(manager, tokens)
        deadline = time.monotonic() + 10
        while manager.persistence_step(force=True):
            assert time.monotonic() < deadline
            time.sleep(.001)
        assert manager.snapshot_stats()["persistence"]["saves"] == 1
        current = (tmp_path / "current.json").read_bytes()
        # A storage-policy version must participate in deployment identity;
        # rejection happens before mx.load and does not mutate the old manifest.
        mismatched.configure_persistence(tmp_path,
            {**fingerprint, "storage_policy": {"growth": False, "pingpong": True}}, 300)
        with monkeypatch.context() as patch:
            patch.setattr(mx, "load", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("loaded wrong identity")))
            assert mismatched.restore_persisted() == 0
        assert mismatched.snapshot_stats()["persistence"]["restore_loads"] == 0
        assert (tmp_path / "current.json").read_bytes() == current

        loaded.configure_persistence(tmp_path, fingerprint, 300)
        assert loaded.restore_persisted() == 1
        assert_independent_readers(loaded, tokens)
        for value, expected in zip(cache_arrays(next(iter(loaded.entries.values())).cache),
                                   cache_arrays(next(iter(manager.entries.values())).cache)):
            assert_bits(value, expected)
    finally:
        for item in (manager, loaded, mismatched):
            item.close_persistence()
