"""CPU-only native/storage contracts retained while preserving stable row batching."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading

import mlx.core as mx
import numpy as np
import pytest

from omlx.patches.deepseek_v41 import storage
from test_deepseek_v41_engram_io import table


@pytest.fixture(autouse=True)
def cpu_and_small_cache(monkeypatch):
    mx.set_default_device(mx.cpu)
    monkeypatch.setattr(storage, "HOT_ROWS_BYTES", 1 << 20)


def test_native_artifact_is_qualified_and_loaded():
    lib = storage._native_gather()
    assert lib is not None, "Fresh source-pinned native library must be built before qualification"
    status = storage.runtime_status()
    assert status["native_library_loaded"] is True
    assert Path(status["library_path"]).parent == Path(storage.__file__).parent
    assert status["source_sha256"][:16] in Path(status["library_path"]).name
    assert len(status["library_sha256"]) == 64


@pytest.mark.parametrize("cached", [False, True])
def test_native_raw_rows_match_stable_selected_read_for_unsorted_duplicates(table, cached):
    reader = storage.TensorFile(table)
    rows = np.array([4999, 0, 40, 40, 1, 4950, 63, 256, 7], np.int64)
    try:
        for key in ("weight", "scale"):
            actual, dtype = reader.gather_rows(key, rows, cached=cached)
            expected, reference_dtype = reader.read(key, rows)
            assert dtype == reference_dtype
            np.testing.assert_array_equal(actual, expected)
    finally:
        reader.close()


@pytest.mark.parametrize("rows", [[0, -1, 4], [0, 5000, 4], [5000]])
def test_native_bounds_check_all_unsorted_ids(table, rows):
    reader = storage.TensorFile(table)
    try:
        with pytest.raises(IndexError, match="outside table"):
            reader.gather_rows("weight", rows)
    finally:
        reader.close()


def test_missing_native_reader_uses_stable_selected_row_fallback(table, monkeypatch):
    monkeypatch.setattr(storage, "_native_gather", lambda: None)
    monkeypatch.setattr(storage, "NATIVE_MIN_ROWS", 1)
    calls = []
    original = storage._read_row_runs

    def batched(*args):
        calls.append(True)
        return original(*args)

    monkeypatch.setattr(storage, "_read_row_runs", batched)
    embed = storage.DiskEngramEmbedding(table, "weight", "scale")
    ids = np.array([0, 4999, 0, 4, 3, 1000], np.int64)
    try:
        actual = embed.gather(ids)
        expected = embed._read_rows(ids)
        assert calls  # Stable positional-run fallback remains reachable.
        for (left, dtype), (right, expected_dtype) in zip(actual, expected):
            assert dtype == expected_dtype
            np.testing.assert_array_equal(left, right)
    finally:
        embed.close()


def test_native_reader_close_waits_for_inflight_bytes(table, monkeypatch):
    native = storage._native_gather()
    entered, release, closing = threading.Event(), threading.Event(), threading.Event()

    class HeldReader:
        @staticmethod
        def ds41_engram_gather(*args):
            entered.set()
            assert release.wait(5)
            return native.ds41_engram_gather(*args)

    monkeypatch.setattr(storage, "_native_gather", lambda: HeldReader())
    reader = storage.TensorFile(table)
    expected, _ = reader.read("weight", np.array([1, 100, 4999]))

    def close():
        closing.set()
        reader.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        future = pool.submit(reader.gather_rows, "weight", np.array([1, 100, 4999]))
        assert entered.wait(5)
        closed = pool.submit(close)
        assert closing.wait(5)
        assert not closed.done()
        release.set()
        actual, _ = future.result(timeout=5)
        closed.result(timeout=5)
    np.testing.assert_array_equal(actual, expected)
    assert reader._direct_fd is reader._cached_fd is reader._mapping is None
    with pytest.raises(RuntimeError, match="closed"):
        reader.gather_rows("weight", [1])


def test_native_read_failure_releases_inflight_ownership(table, monkeypatch):
    class FailedReader:
        @staticmethod
        def ds41_engram_gather(*args):
            return 5  # EIO, as returned for a truncated positional read.

    monkeypatch.setattr(storage, "_native_gather", lambda: FailedReader())
    reader = storage.TensorFile(table)
    with pytest.raises(OSError, match="read failed"):
        reader.gather_rows("weight", [1, 10])
    assert reader._io_active == 0
    reader.close()


def test_oversized_single_lookahead_never_exceeds_queue_budget(table, monkeypatch):
    embed = storage.DiskEngramEmbedding(table, "weight", "scale")
    prefetch = storage.EngramPrefetch()
    monkeypatch.setattr(storage, "LOOKAHEAD_MAX_BYTES", embed.selected_bytes(10))
    try:
        prefetch.submit(embed, np.arange(24).reshape(1, 1, 24), lookahead=True)
        assert not embed._queue.items
        assert embed.pending_bytes() == 0
    finally:
        prefetch.close()
        embed.close()


def test_runtime_status_reports_real_cache_activity_and_bounds(table, monkeypatch):
    monkeypatch.setattr(storage, "NATIVE_MIN_ROWS", 1)
    embed = storage.DiskEngramEmbedding(table, "weight", "scale")
    ids = np.arange(400, dtype=np.int64)
    try:
        embed.gather(ids)
        embed.gather(ids)
        status = storage.runtime_status()
        rows = [row for row in status["embeddings"] if row["path"] == str(table)]
        assert rows
        assert all(row["hot"]["bytes"] <= row["hot"]["budget_bytes"] == 1 << 20 for row in rows)
        assert any(row["hot"]["hits"] >= len(ids) for row in rows)
        assert status["stats"]["direct_calls"] > 0
        assert status["stats"]["logical_row_bytes"] > 0
    finally:
        embed.close()
    assert all(row["path"] != str(table) for row in storage.runtime_status()["embeddings"])
