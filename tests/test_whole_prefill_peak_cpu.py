"""Whole-phase admission against the actual CPU workspace planners; no weights."""
from functools import lru_cache
from itertools import product

import pytest

from test_deepseek_v41_native_workspace_cpu import cpp_plan, policy


@pytest.fixture
def bounded_policy(policy, monkeypatch):
    # The descriptor still queries the real allocation planner. Cache its
    # geometry results because many layers share an identical workspace.
    original = policy.ext.packed_sparse_attention_workspace
    monkeypatch.setattr(policy.ext, "packed_sparse_attention_workspace",
                        lru_cache(maxsize=None)(original))
    return policy


@pytest.mark.parametrize("blocks,size", [(0, 0), (1, 8), (2048, 8), (32, 64)])
def test_index_range_bound_covers_scan_radix_and_tile_switches(bounded_policy, blocks, size):
    profile = bounded_policy.native
    actual = bounded_policy.kernels["_workspace"].packed_index_workspace_bytes
    widths = (1, 8, 9, 32, 64, 128, 129, 256, 512, 8192)
    contexts = (1, 511, 512, 1023, 1024, 2048, 4095, 4096, 8192,
                16384, 65535, 65536, 131072, 262144, 524288, 1048576)
    for width, context in product(widths, contexts):
        bound = profile._index_workspace_range_bound(
            width, context, 512, block_count=blocks, block_size=size)
        for queries, keys in product(widths, contexts):
            if queries <= width and keys <= context:
                assert actual(queries, keys, 512, block_count=blocks,
                              block_size=size) <= bound


def test_whole_phase_covers_attention_switch_before_final_context(bounded_policy):
    profile = bounded_policy.native
    earlier = profile.prediction_components(8192, 139264)["transient_bytes"]
    switched = profile.prediction_components(8192, 147456)["transient_bytes"]
    assert earlier > switched  # Exact final-context pricing is not an upper bound.
    maximum = profile.estimate_whole_prefill_transient_bytes(8192, 147456)
    assert maximum >= earlier
    assert maximum >= profile.prediction_components(8192, 147200)["transient_bytes"]
    # Whole-phase pricing does not alter the exact next-chunk estimate.
    assert profile.prediction_components(8192, 147456)["transient_bytes"] == switched


@pytest.mark.parametrize("context", [65536, 147456, 198241, 371057, 1048576])
def test_whole_phase_covers_actual_chunks_short_tails_and_cache_offsets(bounded_policy, context):
    profile = bounded_policy.native
    bound = profile.estimate_whole_prefill_transient_bytes(8192, context)
    points = {context, 4095, 4096, 16384, 65535, 65536, 139264, 147199, 147200, 147201}
    points.update(range(8192, context, 8192))
    for width in (1, 8, 9, 32, 128, 255, 256, 512, 8192):
        for tokens in points:
            if width <= tokens <= context:
                assert profile.prediction_components(width, tokens)["transient_bytes"] <= bound


@pytest.mark.parametrize("width,context", [(1, 2), (8, 9), (32, 1000), (256, 1000),
                                           (8192, 65536), (8192, 1048576)])
def test_fallback_profile_is_also_bounded_without_native_discount(bounded_policy, width, context):
    profile = bounded_policy.fallback
    bound = profile.estimate_whole_prefill_transient_bytes(width, context)
    for queries in {1, min(8, width), width}:
        for tokens in {queries, max(queries, context // 2), context}:
            assert profile.prediction_components(queries, tokens)["transient_bytes"] <= bound


def test_whole_phase_does_not_iterate_prompt_chunks(bounded_policy, monkeypatch):
    profile = bounded_policy.native
    cls = type(profile)
    original = cls.prediction_components
    calls = []

    def counted(self, width, context):
        calls.append((width, context))
        return original(self, width, context)

    monkeypatch.setattr(cls, "prediction_components", counted)
    profile.estimate_whole_prefill_transient_bytes(8192, 1048576)
    assert calls == [(8192, 1048576), (8, 1048576)]
