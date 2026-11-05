"""Sparse exact checkpoints, SSD-only restores, and tail quota with tiny tensors."""

from types import SimpleNamespace

import mlx.core as mx
import numpy as np

from omlx.patches.deepseek_v41 import prefix_cache_l2
from omlx.patches.deepseek_v41.aligned_io import IO_WORKSPACE_BYTES
from omlx.patches.deepseek_v41.prefix_cache import (
    CHUNK,
    ExactPrefixCache,
    cache_arrays,
    for_model,
)
from test_deepseek_v41_prefix_cache import sample
from test_deepseek_v41_prefix_cache_store import FINGERPRINT


def new_manager(directory, *, tail_max_bytes=None):
    manager = ExactPrefixCache(8 * 2**20, policy="fibonacci_tail")
    if tail_max_bytes is not None:
        manager.tail_max_bytes = tail_max_bytes
    manager.configure_l2(directory, FINGERPRINT, 16 * 2**20)
    return manager


def prompt(length):
    return [token % 110000 for token in range(length)]


def record(manager, tokens):
    _, reused, plan = manager.prepare(tokens)
    assert plan is not None
    last = (len(tokens) - 1) // CHUNK * CHUNK
    for start in range(reused, last, CHUNK):
        manager.record(plan, sample(start + CHUNK), start, CHUNK)
        assert plan.valid and plan.next_position == start + CHUNK
    assert plan.next_position == last
    return reused


def ram_positions(manager):
    return sorted(len(entry.tokens) // 4 for entry in manager.entries.values())


def ssd_positions(manager):
    return sorted(row["tokens"] for row in manager._l2.entries.values())


def test_only_candidate_profile_enables_sparse_policy(monkeypatch):
    def model():
        return SimpleNamespace(_config=SimpleNamespace(ced_prefill=True))

    monkeypatch.setenv("DS41_PREFIX_CACHE_GIB", "2")
    monkeypatch.delenv("DS41_EXACT_PREFIX_CHECKPOINT_POLICY", raising=False)
    assert for_model(model()).snapshot_stats()["checkpoint_policy"] == "dense"
    monkeypatch.setenv("DS41_EXACT_PREFIX_CHECKPOINT_POLICY", "fibonacci_tail")
    stats = for_model(model()).snapshot_stats()
    assert stats["checkpoint_policy"] == "fibonacci_tail"
    assert stats["ram_checkpoint_max_tokens"] == 21 * CHUNK
    assert stats["max_bytes"] == 2 * 2**30


def test_fibonacci_keeps_seven_shallow_points_and_one_200k_tail(tmp_path):
    manager = new_manager(tmp_path)
    try:
        assert record(manager, prompt(200000)) == 0
        assert manager.flush_l2(10)
        fixed = [point * CHUNK for point in (1, 2, 3, 5, 8, 13, 21)]
        assert ram_positions(manager) == fixed
        assert ssd_positions(manager) == [*fixed, 24 * CHUNK]
        stats = manager.snapshot_stats()
        assert stats["record_skipped_policy"] == 16
        assert stats["l2"]["tail_entries"] == 1
        assert manager.prepare(prompt(21 * CHUNK))[1] == 13 * CHUNK
        assert manager.prepare(prompt(21 * CHUNK + 1))[1] == 21 * CHUNK
    finally:
        manager.close_l2(timeout_seconds=10)


def test_cold_catalog_misses_are_counted_without_waiting_for_writes(tmp_path, monkeypatch):
    manager = new_manager(tmp_path)
    try:
        tokens = prompt(4 * CHUNK + 1)
        with monkeypatch.context() as patch:
            def wait(*args, **kwargs):
                raise AssertionError("a missing catalog key must not wait for the SSD writer")

            patch.setattr(manager._l2, "_finish", wait)
            assert manager.prepare(tokens)[1] == 0
        assert manager.snapshot_stats()["l2"]["misses"] == 4

        assert record(manager, tokens) == 0
        assert manager.flush_l2(10)
        misses = manager.snapshot_stats()["l2"]["misses"]
        assert manager.prepare(tokens)[1] == 4 * CHUNK
        assert manager.prepare(tokens[:3 * CHUNK + 1])[1] == 3 * CHUNK
        assert manager.snapshot_stats()["l2"]["misses"] == misses
    finally:
        manager.close_l2(timeout_seconds=10)


def test_disk_only_tail_and_deep_fixed_restore_bits_after_restart(tmp_path):
    tail_end = 24 * CHUNK
    fixed_end = 34 * CHUNK
    first = new_manager(tmp_path)
    try:
        assert record(first, prompt(200000)) == 0
        assert record(first, prompt(fixed_end + 1)) == tail_end
        assert first.flush_l2(10)
        assert ram_positions(first)[-1] == 21 * CHUNK
        assert ssd_positions(first)[-2:] == [tail_end, fixed_end]
    finally:
        first.close_l2(timeout_seconds=10)

    reopened = new_manager(tmp_path)
    try:
        for end in (tail_end, fixed_end, fixed_end):
            restored, count, _ = reopened.prepare(prompt(end + 1))
            assert count == end
            assert reopened.bytes == 0 and not reopened.entries
            expected = sample(end)
            for actual, original in zip(cache_arrays(restored), cache_arrays(expected), strict=True):
                assert actual.shape == original.shape and actual.dtype == original.dtype
                assert np.array_equal(np.asarray(actual.view(mx.uint8)),
                                      np.asarray(original.view(mx.uint8)))

        stats = reopened.snapshot_stats()["l2"]
        assert stats["ssd_only_hits"] == 3 and stats["tail_hits"] == 1
        assert reopened.prepare(prompt(21 * CHUNK + 1))[1] == 21 * CHUNK
        assert ram_positions(reopened) == [21 * CHUNK]
    finally:
        reopened.close_l2(timeout_seconds=10)


def branch_prompt(branch):
    return prompt(3 * CHUNK) + [60000 + branch] * CHUNK + [1]


def test_tail_budget_evicts_oldest_tail_without_evicting_fixed_entries(tmp_path):
    manager = new_manager(tmp_path, tail_max_bytes=290000)
    try:
        for branch in (0, 1):
            record(manager, branch_prompt(branch))
            assert manager.flush_l2(10)
        assert manager.prepare(branch_prompt(0))[1] == 4 * CHUNK
        record(manager, branch_prompt(2))
        assert manager.flush_l2(10)

        assert manager.prepare(branch_prompt(0))[1] == 4 * CHUNK
        assert manager.prepare(branch_prompt(1))[1] == 3 * CHUNK
        assert manager.prepare(branch_prompt(2))[1] == 4 * CHUNK
        assert ssd_positions(manager) == [point * CHUNK for point in (1, 2, 3, 4, 4)]
        stats = manager.snapshot_stats()["l2"]
        assert stats["tail_entries"] == 2
        assert 0 < stats["tail_disk_bytes"] <= stats["tail_max_bytes"] == 290000
        assert stats["evictions"] == 1
    finally:
        manager.close_l2(timeout_seconds=10)


def test_tail_cannot_evict_fixed_entries_to_make_global_space(tmp_path):
    manager = new_manager(tmp_path)
    try:
        record(manager, prompt(3 * CHUNK + 1))
        assert manager.flush_l2(10)
        before = manager.snapshot_stats()["l2"]
        manager.set_l2_max_bytes(before["disk_bytes"] + 32 * 2**10, timeout_seconds=10)
        assert record(manager, branch_prompt(0)) == 3 * CHUNK
        assert manager.flush_l2(10)
        assert ssd_positions(manager) == [point * CHUNK for point in (1, 2, 3)]
        assert manager.snapshot_stats()["l2"]["evictions"] == before["evictions"]
    finally:
        manager.close_l2(timeout_seconds=10)


def test_tail_guard_or_corruption_falls_back_to_shallow_prefix(tmp_path):
    manager = new_manager(tmp_path)
    try:
        tokens = branch_prompt(0)
        record(manager, tokens)
        assert manager.flush_l2(10)
        before = manager.bytes
        manager.ensure_headroom = lambda size: size <= IO_WORKSPACE_BYTES + 7 * CHUNK * 4
        assert manager.prepare(tokens)[1] == 3 * CHUNK
        assert manager.bytes == before
        assert manager.snapshot_stats()["l2"]["skipped_memory"] == 1
        assert ssd_positions(manager)[-1] == 4 * CHUNK

        manager.ensure_headroom = None
        tail = next(row for row in manager._l2.entries.values() if row["tokens"] == 4 * CHUNK)
        with (tmp_path / (tail["key"] + ".safetensors")).open("r+b") as stream:
            stream.seek(-1, 2)
            value = stream.read(1)
            stream.seek(-1, 2)
            stream.write(bytes([value[0] ^ 1]))
        assert manager.prepare(tokens)[1] == 3 * CHUNK
        assert manager.snapshot_stats()["l2"]["read_failures"] == 1
        assert ssd_positions(manager)[-1] == 3 * CHUNK
    finally:
        manager.close_l2(timeout_seconds=10)


def test_failed_tail_write_never_becomes_a_cache_hit(tmp_path, monkeypatch):
    manager = new_manager(tmp_path)
    real_write = prefix_cache_l2._write_checkpoint

    def write(path, header, chunks, descriptor):
        if descriptor["tokens"] == 4 * CHUNK:
            raise OSError("injected tail write failure")
        return real_write(path, header, chunks, descriptor)

    monkeypatch.setattr(prefix_cache_l2, "_write_checkpoint", write)
    try:
        tokens = branch_prompt(0)
        record(manager, tokens)
        assert not manager.flush_l2(10)
        assert manager.prepare(tokens)[1] == 3 * CHUNK
        assert manager.snapshot_stats()["l2"]["write_failures"] == 1
        assert ssd_positions(manager) == ram_positions(manager)
        assert not list(tmp_path.glob("*.pending"))
    finally:
        manager.close_l2(timeout_seconds=10)
