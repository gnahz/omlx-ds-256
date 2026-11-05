"""CPU/mock regression for exact-CED cache admission; never load a model."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from omlx.exceptions import PrefillMemoryExceededError
from omlx.request import Request, SamplingParams
from omlx.scheduler import Scheduler, SchedulerConfig


GIB = 1024**3
TOKENS = 139_570
CURRENT = int(232.05 * GIB)
PEAK = int(5.40 * GIB)


@pytest.fixture
def setup_guard(monkeypatch):
    """Real scheduler decisions with controlled process/allocator readings."""
    model = MagicMock()
    model.layers = []
    model.config = SimpleNamespace(num_hidden_layers=40, num_key_value_heads=1,
                                   num_attention_heads=64, head_dim=512)
    del model.make_cache
    tokenizer = MagicMock()
    tokenizer.eos_token_id = 2
    scheduler = Scheduler(model=model, tokenizer=tokenizer, config=SchedulerConfig(
        max_num_seqs=8, prefill_step_size=8192, paged_cache_block_size=0,
        prefill_peak_target_gib=236))
    scheduler._prefill_memory_guard = True
    scheduler._memory_hard_limit_bytes = 240 * GIB
    scheduler._memory_abort_limit_bytes = 240 * GIB
    scheduler._memory_hard_watermark_bytes = int(240 * GIB * .999)
    scheduler._prefill_headroom_safety = .995
    scheduler._prefill_abort_margin = .995
    scheduler._prefill_min_chunk_tokens = 32
    scheduler._memory_guard_tier = "custom"
    scheduler._memory_dynamic_ceiling_bytes = 240 * GIB
    state = {"current": CURRENT, "transient": PEAK - TOKENS * 890,
             "events": [], "released": 0, "evictable": 2 * GIB}
    scheduler._current_usage_bytes = lambda **kwargs: state["current"]
    scheduler.memory_monitor.estimate_resident_kv_bytes = MagicMock(
        side_effect=lambda count, **kwargs: count * 890)
    scheduler._admission_transient_bound = lambda *args, **kwargs: state["transient"]
    scheduler._qwen4_text_gathered_pricing = lambda **kwargs: False
    scheduler._resolve_generation_prompt_start = lambda request: None
    scheduler._releasable_hot_cache_credit = lambda required: 0

    def clear_cache(self):
        state["events"].append("clear_pool")

    monkeypatch.setattr(Scheduler, "_clear_cache", clear_cache)

    def evict(required):
        state["events"].append(("evict_exact", required))
        released = min(required, state["evictable"])
        state["current"] -= released
        state["evictable"] -= released
        state["released"] += released
        exact.bytes -= released
        return released

    exact = SimpleNamespace(max_bytes=2 * GIB, bytes=2 * GIB, evict_bytes=evict)

    def prepare(tokens, step, **kwargs):
        state["events"].append("lookup")
        assert step == 8192
        return None, 0, SimpleNamespace()

    exact.prepare = MagicMock(side_effect=prepare)
    model._omlx_exact_prefix_cache = exact
    return scheduler, state, exact


def request():
    req = Request(request_id="original-139570", prompt=list(range(TOKENS)),
                  sampling_params=SamplingParams(max_tokens=64))
    req.prompt_token_ids = list(range(TOKENS))
    req.num_prompt_tokens = TOKENS
    # The normal async idle-model retry remains available in production.
    # Final-rejection tests model the state after those bounded attempts.
    req.prefill_eviction_retries = 2
    return req


def test_recorded_rejection_waits_for_exact_lookup_and_bounded_reclaim(setup_guard):
    """Old add_request rejects237.45>236 before lookup can reclaim1.45GiB."""
    scheduler, state, exact = setup_guard
    req = request()
    assert scheduler._admission_estimate(num_prompt_tokens=TOKENS, cached_tokens=0,
                                         current=CURRENT).estimated == CURRENT + PEAK
    scheduler.add_request(req)
    assert req in scheduler.waiting
    assert state["events"] == []
    scheduler._prepare_prefix_cache_for_request(req)
    assert scheduler._preflight_memory_check(req) is None
    assert state["events"][0] == "lookup"
    assert state["events"][1] == "clear_pool"
    assert state["events"][2] == ("evict_exact", CURRENT + PEAK - 236 * GIB)
    assert 0 < state["released"] < 2 * GIB
    exact.prepare.assert_called_once()


def test_exact_cache_miss_keeps_real_over_budget_rejection(setup_guard):
    scheduler, state, _ = setup_guard
    req = request()
    state["evictable"] = 0  # All checkpoint arrays are still pinned/active.
    scheduler.add_request(req)
    scheduler._prepare_prefix_cache_for_request(req)
    rejection = scheduler._preflight_memory_check(req)
    assert rejection is not None
    assert rejection.estimated_bytes == CURRENT + PEAK
    assert rejection.limit_bytes == 236 * GIB
    assert state["events"][0] == "lookup"
    assert req.cached_tokens == 0


@pytest.mark.parametrize("usage,accepted", [(CURRENT, True), (int(232.61 * GIB), False)])
def test_238_target_keeps_two_gib_margin_and_rejects_real_overage(setup_guard, usage, accepted):
    scheduler, state, _ = setup_guard
    scheduler.config.prefill_peak_target_gib = 238
    state["current"] = usage
    state["evictable"] = 0
    req = request()
    scheduler.add_request(req)
    scheduler._prepare_prefix_cache_for_request(req)
    assert scheduler._admission_limit_bytes() == 238 * GIB
    rejection = scheduler._preflight_memory_check(req)
    assert (rejection is None) is accepted
    if rejection is not None:
        assert rejection.limit_bytes == 238 * GIB
        assert "configured prefill target 238.00 GB" in rejection.message
        assert "Raise custom_ceiling_bytes" not in rejection.message


def test_no_exact_cache_still_checks_immediate_preflight(setup_guard):
    scheduler, _, _ = setup_guard
    del scheduler.model._omlx_exact_prefix_cache
    with pytest.raises(PrefillMemoryExceededError):
        scheduler.add_request(request())
    assert not scheduler.waiting
    assert not scheduler.requests


def test_reclaim_reprices_returning_flat_overhead_instead_of_crediting_pool_twice(setup_guard, monkeypatch):
    scheduler, state, _ = setup_guard
    req = request()
    state["evictable"] = 0
    scheduler.add_request(req)
    scheduler._prepare_prefix_cache_for_request(req)
    calls = 0

    def clear_cache(self):
        nonlocal calls
        calls += 1
        state["events"].append("clear_pool")
        if calls == 1:
            # A released allocator pool becomes a future reallocation charge.
            state["current"] -= 2 * GIB
            state["transient"] += 2 * GIB

    monkeypatch.setattr(Scheduler, "_clear_cache", clear_cache)
    rejection = scheduler._preflight_memory_check(req)
    assert rejection is not None
    assert rejection.estimated_bytes == CURRENT + PEAK


def test_lower_live_physical_limit_still_clips_238_target(setup_guard):
    scheduler, _, _ = setup_guard
    scheduler.config.prefill_peak_target_gib = 238
    scheduler._memory_hard_limit_bytes = 234 * GIB
    assert scheduler._admission_limit_bytes() < 234 * GIB


def test_exact_lookup_prices_only_the_validated_residual(setup_guard):
    scheduler, state, exact = setup_guard
    scheduler.config.prefill_peak_target_gib = 238
    prefix = 131_072
    exact.prepare.side_effect = lambda *args, **kwargs: (None, prefix, SimpleNamespace())
    req = request()
    scheduler.add_request(req)
    scheduler._prepare_prefix_cache_for_request(req)
    assert scheduler._preflight_memory_check(req) is None
    assert req.cached_tokens == prefix
    assert len(req.remaining_tokens) == TOKENS - prefix
    scheduler.memory_monitor.estimate_resident_kv_bytes.assert_called_once_with(
        TOKENS - prefix, chunk_tokens=32)
    assert state["released"] == 0


def test_logical_eviction_is_not_credited_until_physical_usage_settles(setup_guard):
    scheduler, state, exact = setup_guard
    req = request()

    def release_only_python_references(required):
        # The checkpoint owner releases bytes but the kernel still charges
        # them. Only measured usage may authorize the subsequent allocation.
        exact.bytes = 0
        return min(required, 2 * GIB)

    exact.evict_bytes = release_only_python_references
    scheduler.add_request(req)
    scheduler._prepare_prefix_cache_for_request(req)
    rejection = scheduler._preflight_memory_check(req)
    assert rejection is not None
    assert rejection.estimated_bytes == CURRENT + PEAK
    assert exact.bytes == 0


def test_returning_pool_overhead_is_repriced_before_bounded_exact_eviction(setup_guard, monkeypatch):
    scheduler, state, exact = setup_guard
    req = request()
    calls = 0

    def clear_cache(self):
        nonlocal calls
        calls += 1
        state["events"].append("clear_pool")
        if calls == 1:
            state["current"] -= 2 * GIB
            state["transient"] += 2 * GIB

    monkeypatch.setattr(Scheduler, "_clear_cache", clear_cache)
    scheduler.add_request(req)
    scheduler._prepare_prefix_cache_for_request(req)
    assert scheduler._preflight_memory_check(req) is None
    assert state["released"] == CURRENT + PEAK - 236 * GIB
    assert state["events"] == ["lookup", "clear_pool",
                               ("evict_exact", state["released"]), "clear_pool"]
