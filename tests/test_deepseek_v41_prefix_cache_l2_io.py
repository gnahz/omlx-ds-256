"""Small exact-L2 round trips through aligned disk I/O, run with MLX on CPU."""

import numpy as np
import mlx.core as mx

from omlx.patches.deepseek_v41.prefix_cache import ExactPrefixCache, cache_arrays
from test_deepseek_v41_prefix_cache_store import add, FINGERPRINT


def manager(directory):
    result = ExactPrefixCache(2**20)
    result.configure_l2(directory, FINGERPRINT, 4 * 2**20)
    return result


def test_aligned_l2_restores_exact_bits_after_ram_eviction(tmp_path):
    first = manager(tmp_path)
    try:
        tokens = add(first, 16384)
        before, count, _ = first.prepare(tokens)
        assert count == 16384 and first.flush_l2(10)
        written = first.snapshot_stats()["l2"]
        assert written["writes"] == 2 and written["entries"] == 2
        first.clear()
        after, count, _ = first.prepare(tokens)
        assert count == 16384
        assert first.snapshot_stats()["l2"]["hits"] == 1
        for left, right in zip(cache_arrays(before), cache_arrays(after)):
            assert left.dtype == right.dtype and left.shape == right.shape
            assert np.array_equal(np.asarray(left.view(mx.uint8)), np.asarray(right.view(mx.uint8)))
    finally:
        first.close_l2(timeout_seconds=10)


def test_aligned_l2_reopens_existing_catalog_and_rejects_corrupt_payload(tmp_path):
    first = manager(tmp_path)
    tokens = add(first)
    assert first.flush_l2(10)
    first.close_l2(timeout_seconds=10)
    reopened = manager(tmp_path)
    try:
        assert reopened.prepare(tokens)[1] == 8192
        reopened.clear()
        row = next(iter(reopened._l2.entries.values()))
        path = reopened._l2.directory / (row["key"] + ".safetensors")
        with path.open("r+b") as stream:
            stream.seek(-1, 2)
            value = stream.read(1)
            stream.seek(-1, 2)
            stream.write(bytes([value[0] ^ 1]))
        assert reopened.prepare(tokens)[1] == 0
        assert reopened.snapshot_stats()["l2"]["read_failures"] == 1
    finally:
        reopened.close_l2(timeout_seconds=10)
