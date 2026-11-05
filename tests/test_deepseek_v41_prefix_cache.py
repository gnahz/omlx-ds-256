"""Exact CED cache ownership, geometry and eviction (small arrays only)."""

import numpy as np
import mlx.core as mx
import pytest
from omlx.patches.deepseek_v41.cache import DeepseekV41Cache
from omlx.patches.deepseek_v41.prefix_cache import ExactPrefixCache, compact_copy
from omlx.patches.mlx_lm_mtp.deepseek_v4_dspark import (
    DSparkContextCache,
    _DSparkPrimeContext,
)


def sample(end=8192):
    c = DeepseekV41Cache(2)
    c.cache = [
        mx.array([end]),
        mx.arange(24).reshape(1, 8, 3),
        mx.zeros((1, 4, 4), mx.uint32),
        mx.zeros((1, 4, 4), mx.uint32),
        mx.zeros((1, 0, 4)),
        mx.zeros((1, 0, 4)),
        mx.array([[1, 2]], mx.int64),
    ]
    stage = DSparkContextCache(128)
    stage.keys = mx.arange(24).reshape(1, 1, 8, 3)
    stage.offset = end
    c._omlx_mtp_prime_ctx = _DSparkPrimeContext([stage], end)
    return [c]


def test_disabled_persistence_reports_observable_zero_io_counters():
    manager = ExactPrefixCache(2**20)
    expected = {"enabled": False, "disk_reads": 0, "restored_checkpoints": 0}
    assert manager.snapshot_stats()["persistence"] == expected
    tokens = list(range(9000))
    _, _, plan = manager.prepare(tokens)
    manager.record(plan, sample(), 0, 8192)
    assert manager.prepare(tokens)[1] == 8192
    assert manager.restore_persisted() == 0
    assert manager.persistence_step(force=True) is False
    assert manager.persistence_pending is False
    manager.clear()
    assert manager.snapshot_stats()["persistence"] == expected
    assert manager._persistence is None


def test_enabled_persistence_retains_store_counters():
    from types import SimpleNamespace
    manager = ExactPrefixCache(2**20)
    counters = {"enabled": True, "disk_reads": 13, "restored_checkpoints": 7,
                "saves": 3, "pending": False}
    manager._persistence = SimpleNamespace(stats=lambda: counters)
    assert manager.snapshot_stats()["persistence"] is counters


def test_snapshot_and_two_readers_own_python_and_array_state():
    cache = sample()
    tokens = list(range(17000))
    manager = ExactPrefixCache(2**20)
    _, _, plan = manager.prepare(tokens)
    manager.record(plan, cache, 0, 8192)
    a, n, _ = manager.prepare(tokens)
    b, _, _ = manager.prepare(tokens)
    assert n == 8192
    cache[0][1][:] = 99
    cache[0]._omlx_mtp_prime_ctx.caches[0].keys[:] = 88
    a[0][1][:] = 77
    a[0]._omlx_mtp_prime_ctx.caches[0].keys[:] = 66
    a[0]._omlx_mtp_prime_ctx.caches[0].offset = 9
    assert b[0][1][0, 0, 0].item() == 0
    assert b[0]._omlx_mtp_prime_ctx.caches[0].keys[0, 0, 0, 0].item() == 0
    assert b[0]._omlx_mtp_prime_ctx.caches[0].offset == 8192


def test_partial_chunk_and_changed_geometry_never_create_checkpoint():
    manager = ExactPrefixCache(2**20)
    tokens = list(range(17000))
    _, _, plan = manager.prepare(tokens)
    manager.record(plan, sample(), 0, 256)
    manager.record(plan, sample(), 256, 8192)
    assert not manager.entries and not plan.valid
    assert manager.prepare(tokens, 4096) == (None, 0, None)


def test_longest_match_divergence_and_leave_final_token():
    manager = ExactPrefixCache(2**20)
    tokens = list(range(17000))
    _, _, plan = manager.prepare(tokens)
    manager.record(plan, sample(), 0, 8192)
    manager.record(plan, sample(16384), 8192, 8192)
    assert manager.prepare(tokens)[1] == 16384
    assert manager.prepare(tokens[:8192])[1] == 0
    assert manager.prepare(tokens[:16384])[1] == 8192
    changed = tokens.copy()
    changed[9000] = 123
    assert manager.prepare(changed)[1] == 8192
    changed[100] = 456
    assert manager.prepare(changed)[1] == 0


def test_eviction_hard_budget_and_disabled_cache():
    manager = ExactPrefixCache(100000)
    tokens = list(range(33000))
    _, _, plan = manager.prepare(tokens)
    for start in range(0, 32768, 8192):
        manager.record(plan, sample(start + 8192), start, 8192)
        assert manager.bytes <= manager.max_bytes
    assert manager.evictions > 0
    assert manager.prepare(tokens)[1] == 24576  # 32K snapshot exceeds budget
    manager.clear()
    assert manager.bytes == 0 and not manager.entries
    manager.max_bytes = 0
    assert manager.prepare(tokens) == (None, 0, None)


def test_compact_copy_preserves_raw_float_payloads_and_slice():
    bits = np.array(
        [0, 0x80000000, 0x7FC00001, 0xFFC00002, 0x7F800000, 1, 0xFFFFFFFF],
        dtype=np.uint32,
    )
    copied = compact_copy(mx.array(bits).view(mx.float32))
    assert np.array_equal(np.asarray(copied.view(mx.uint32)), bits)
    x = mx.arange(10000)
    y = compact_copy(x[512:1024])
    mx.eval(y)
    x[:] = 0
    assert y.tolist() == list(range(512, 1024))


def test_model_ownership_and_draft_mode_change(monkeypatch):
    from types import SimpleNamespace
    from omlx.patches.deepseek_v41.prefix_cache import for_model

    monkeypatch.setenv("DS41_PREFIX_CACHE_GIB", "2")
    first = SimpleNamespace(
        _config=SimpleNamespace(ced_prefill=True),
        _omlx_dspark_decode_enabled=True,
        _omlx_mtp_depth=4,
    )
    second = SimpleNamespace(
        _config=SimpleNamespace(ced_prefill=True),
        _omlx_dspark_decode_enabled=True,
        _omlx_mtp_depth=4,
    )
    manager = for_model(first)
    assert for_model(first) is manager
    assert for_model(second) is not manager
    first._omlx_mtp_depth = 3
    assert for_model(first) is not manager
    first._config.ced_prefill = False
    assert for_model(first) is None
    monkeypatch.setenv("DS41_PREFIX_CACHE_GIB", "0")
    assert for_model(second) is None


def test_memory_guard_runs_before_snapshot_or_restore_and_tracks_skips(monkeypatch):
    import omlx.patches.deepseek_v41.prefix_cache as module

    manager = ExactPrefixCache(2**20)
    tokens = list(range(9000))
    _, _, plan = manager.prepare(tokens)
    manager.ensure_headroom = lambda size: False
    with monkeypatch.context() as patch:
        patch.setattr(module, "clone_cache", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("allocated")))
        manager.record(plan, sample(), 0, 8192)
    assert not manager.entries and manager.skipped_memory == 1
    manager.ensure_headroom = None
    _, _, plan = manager.prepare(tokens)
    manager.record(plan, sample(), 0, 8192)
    manager.ensure_headroom = lambda size: False
    assert manager.prepare(tokens)[1] == 0
    assert manager.snapshot_stats()["skipped_memory"] == 2
    assert manager.snapshot_stats()["reused_tokens"] == 0


def test_restore_pins_selected_snapshot_while_memory_guard_evicts():
    manager = ExactPrefixCache(2**20)
    tokens = list(range(17000))
    _, _, plan = manager.prepare(tokens)
    manager.record(plan, sample(), 0, 8192)
    manager.record(plan, sample(16384), 8192, 8192)
    manager.ensure_headroom = lambda size: bool(manager.evict_bytes(size))
    assert manager.prepare(tokens)[1] == 16384
    assert len(manager.entries) == 1
    assert manager.snapshot_stats()["reused_tokens"] == 16384


DOMAIN, IMAGE_A, IMAGE_B, IMAGE_C = (character * 64 for character in "dabc")


def image_tokens(length=17000, positions=(100, 9000)):
    tokens = list(range(length))
    for position in positions:
        tokens[position] = 129264
    return tokens


def image_identity(ranges):
    return dict(extra_keys=ranges[-1][1], extra_key_token_start=ranges[0][0],
                extra_key_ranges=ranges)


def record_image(result, tokens, ranges):
    _, _, plan = result.prepare(tokens, **image_identity(ranges))
    assert plan is not None
    for start in range(plan.next_position, len(tokens) - 8192, 8192):
        result.record(plan, sample(start + 8192), start, 8192)


def test_image_identity_same_image_hits_and_same_token_ids_other_image_misses():
    result = ExactPrefixCache(2**20)
    tokens = image_tokens(9000, (100,))
    ranges = [(0, (DOMAIN,)), (100, (IMAGE_A,))]
    record_image(result, tokens, ranges)
    assert result.prepare(tokens, **image_identity(ranges))[1] == 8192
    other = [(0, (DOMAIN,)), (100, (IMAGE_B,))]
    assert result.prepare(tokens, **image_identity(other))[1] == 0
    assert result.prepare(tokens) == (None, 0, None)


def test_changed_late_image_or_appended_image_preserves_earlier_prefix():
    result = ExactPrefixCache(2**20)
    tokens = image_tokens()
    ranges = [(0, (DOMAIN,)), (100, (IMAGE_A,)), (9000, (IMAGE_B,))]
    record_image(result, tokens, ranges)
    assert result.prepare(tokens, **image_identity(ranges))[1] == 16384
    changed = [(0, (DOMAIN,)), (100, (IMAGE_A,)), (9000, (IMAGE_C,))]
    assert result.prepare(tokens, **image_identity(changed))[1] == 8192
    extended = tokens + list(range(17000, 26000))
    extended[20000] = 129264
    appended = ranges + [(20000, (IMAGE_C,))]
    assert result.prepare(extended, **image_identity(appended))[1] == 16384
    incomplete = image_identity(ranges)
    incomplete["extra_key_ranges"] = ranges[:-1]
    assert result.prepare(tokens, **incomplete) == (None, 0, None)


@pytest.mark.parametrize("start,hit", [(8191, 0), (8192, 8192), (8193, 8192)])
def test_image_identity_at_exact_boundary_applies_only_to_following_chunk(start, hit):
    result = ExactPrefixCache(2**20)
    tokens = image_tokens(17000, (start,))
    ranges = [(0, (DOMAIN,)), (start, (IMAGE_A,))]
    record_image(result, tokens, ranges)
    changed = [(0, (DOMAIN,)), (start, (IMAGE_B,))]
    assert result.prepare(tokens, **image_identity(changed))[1] == hit


def test_prefill_domain_separates_identical_text_prefix_from_vlm_path():
    result = ExactPrefixCache(2**20)
    tokens = image_tokens(17000, (9000,))
    _, _, plan = result.prepare(tokens[:9000])
    result.record(plan, sample(), 0, 8192)
    ranges = [(0, (DOMAIN,)), (9000, (IMAGE_A,))]
    assert result.prepare(tokens, **image_identity(ranges))[1] == 0
    record_image(result, tokens, ranges)
    assert result.prepare(tokens, **image_identity(ranges))[1] == 16384
    assert result.prepare(tokens[:9000])[1] == 8192


def test_exact_image_proof_is_compared_after_hash_lookup():
    result = ExactPrefixCache(2**20)
    tokens = image_tokens(9000, (100,))
    ranges = [(0, (DOMAIN,)), (100, (IMAGE_A,))]
    record_image(result, tokens, ranges)
    entry = next(iter(result.entries.values()))
    entry.identity = ((0, (DOMAIN,)), (100, (IMAGE_B,)))
    assert result.prepare(tokens, **image_identity(ranges))[1] == 0


@pytest.mark.parametrize("kwargs", [
    {}, {"extra_keys": ()}, {"extra_keys": "a" * 64},
    {"extra_keys": ("not-a-content-hash",)},
    {"extra_key_token_start": 0},
    {"extra_key_ranges": []},
    {"extra_key_ranges": [(0, (IMAGE_A,)), (-1, (IMAGE_B,))]},
    {"extra_key_ranges": [(0, (IMAGE_A,)), (0, (IMAGE_B,))]},
    {"extra_key_ranges": [(101, (IMAGE_A,))]},
    {"extra_key_ranges": [(0, (IMAGE_A,)), (9000, (IMAGE_B,))]},
    {"extra_key_ranges": [(False, (IMAGE_A,))]},
    {"extra_keys": (IMAGE_B,), "extra_key_ranges": [(0, (IMAGE_A,))]},
    {"extra_key_token_start": 1, "extra_key_ranges": [(0, (IMAGE_A,))]},
])
def test_missing_or_malformed_image_provenance_never_becomes_text_cache(kwargs):
    result = ExactPrefixCache(2**20)
    assert result.prepare(image_tokens(9000, (100,)), **kwargs) == (None, 0, None)
    assert not result.entries


def test_whole_request_image_hash_fallback_is_conservative():
    result = ExactPrefixCache(2**20)
    tokens = image_tokens(9000, (100,))
    kwargs = dict(extra_keys=(IMAGE_A,), extra_key_token_start=0)
    _, _, plan = result.prepare(tokens, **kwargs)
    result.record(plan, sample(), 0, 8192)
    assert result.prepare(tokens, **kwargs)[1] == 8192
    assert result.prepare(tokens, extra_keys=(IMAGE_B,))[1] == 0


@pytest.mark.parametrize("dtype,bits", [
    (mx.float32, [0, 0x80000000, 0x7FC00001, 0xFFC00002, 1, 0xFFFFFFFF]),
    (mx.bfloat16, [0, 0x8000, 0x7FC1, 0xFFC2, 1, 0xFFFF]),
    (mx.float16, [0, 0x8000, 0x7E01, 0xFE02, 1, 0xFFFF]),
    (mx.uint32, [0, 1, 0xFFFFFFFF, 0x80000000, 7, 999]),
])
def test_cpu_compact_copy_strides_bits_and_independent_ownership(dtype, bits, monkeypatch):
    import omlx.patches.deepseek_v41.prefix_cache as module

    def forbidden(*args, **kwargs):
        pytest.fail("CPU copy must never create a Metal kernel")

    monkeypatch.setattr(mx.fast, "metal_kernel", forbidden)
    with mx.stream(mx.cpu):
        kind = mx.uint32 if dtype in (mx.float32, mx.uint32) else mx.uint16
        base = mx.array(bits * 4, dtype=kind).view(dtype).reshape(4, 6)
        view = base[::2, ::-2]
        expected = np.asarray(mx.contiguous(view).view(mx.uint8)).copy()
        copied = module.compact_copy(view)
        assert copied.dtype == view.dtype and copied.shape == view.shape
        np.testing.assert_array_equal(np.asarray(copied.view(mx.uint8)), expected)
        base[:] = 0
        np.testing.assert_array_equal(np.asarray(copied.view(mx.uint8)), expected)
        copied[:] = 1
        assert mx.all(base == 0).item()
