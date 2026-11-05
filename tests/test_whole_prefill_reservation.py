"""Whole-prefill ownership and fixed execution width, using byte counters only."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import omlx.scheduler as scheduler_module
from omlx.prefill_progress import get_prefill_tracker
from omlx.request import Request, SamplingParams
from omlx.request_memory_budget import GIB, RequestMemoryEnvelope
from omlx.scheduler import Scheduler, SchedulerConfig, _PrefillBudgetWait


@pytest.fixture
def whole_scheduler(monkeypatch):
    monkeypatch.setenv("OMLX_PIPELINE_MEMORY_BUDGET", "1")
    monkeypatch.setenv("OMLX_PREFILL_WHOLE_RESERVATION", "1")
    model, tokenizer = MagicMock(), MagicMock()
    model.layers = []
    model.config = SimpleNamespace(
        num_hidden_layers=40, num_key_value_heads=1,
        num_attention_heads=64, head_dim=512,
    )
    del model.make_cache
    tokenizer.eos_token_id = 2
    scheduler = Scheduler(model, tokenizer, SchedulerConfig(
        max_num_seqs=4, paged_cache_block_size=0,
        prefill_step_size=8192, prefill_peak_target_gib=238,
    ))
    physical = {"current": 220 * GIB, "free": 16 * GIB}
    monkeypatch.setattr(scheduler_module, "get_phys_footprint", lambda: physical["current"])
    monkeypatch.setattr(scheduler_module, "get_macos_vm_stats", lambda: {"free": physical["free"]})
    scheduler._current_usage_bytes = lambda **kw: physical["current"]
    scheduler._memory_hard_limit_bytes = 240 * GIB
    scheduler._memory_abort_limit_bytes = 240 * GIB
    scheduler._memory_hard_watermark_bytes = int(239.8 * GIB)
    scheduler._prefill_headroom_safety = scheduler._prefill_abort_margin = .995
    scheduler._memory_limit_bytes = 232 * GIB
    scheduler._prefill_min_chunk_tokens = 256
    scheduler._prefill_preferred_chunk_tokens = 512
    scheduler.memory_monitor.prefill_prediction_components = lambda *a: {"profile": "deepseek_v41"}
    scheduler.memory_monitor.estimate_resident_kv_bytes = lambda count, **kw: count * 890
    scheduler.memory_monitor.estimate_chunk_transient_bytes = lambda q, k: q * k * 20
    scheduler.memory_monitor.estimate_prompt_kv_bytes = lambda count: count * 890
    scheduler._predicted_chunk_transient = lambda q, k, **kw: int((.5 + q / 8192 * 5) * GIB)
    scheduler.memory_monitor.estimate_whole_prefill_transient_bytes = lambda q, context: (
        scheduler._predicted_chunk_transient(q, context - q, include_shared_reclaim=False)
        / scheduler._PREFILL_TRANSIENT_SAFETY
        - scheduler.memory_monitor.estimate_prompt_kv_bytes(q)
    )
    scheduler._admission_transient_bound = scheduler._predicted_chunk_transient
    scheduler._reclaim_prefill_headroom = MagicMock(side_effect=lambda: physical["current"])
    scheduler._prefill_gate_open = lambda: True
    scheduler._should_defer_for_cache_freshness = lambda request: False
    scheduler._store_cache_gate = None
    scheduler._admission_paused = False
    tracker = get_prefill_tracker()
    tracker.clear()
    yield scheduler, physical
    tracker.clear()


def request(scheduler, request_id="whole", tokens=200000, maximum=256000):
    result = Request(request_id=request_id, prompt=[1], sampling_params=SamplingParams(max_tokens=maximum))
    result.num_prompt_tokens = tokens
    scheduler.requests[request_id] = result
    return result


def test_whole_policy_is_loaded_and_reported(whole_scheduler):
    scheduler, _ = whole_scheduler
    assert scheduler._prefill_whole_reservation is True
    policy = scheduler._memory_budget_snapshot()["prefill_policy"]
    assert policy["mode"] == "whole_prefill"
    assert policy["fixed_chunk_tokens"] == 8192
    assert scheduler.config.max_num_seqs == 4


def test_full_prefill_waits_even_when_a_small_chunk_would_fit(whole_scheduler):
    scheduler, physical = whole_scheduler
    physical["free"] = 2 * GIB
    head = request(scheduler)
    budget = scheduler._request_memory_budget
    small = scheduler._pipeline_progress_envelope(head, prefill_step_size=256)
    assert budget.reserve("small-probe", small, physical["current"], scheduler._pipeline_system_limit())
    budget.release("small-probe")

    assert not scheduler._reserve_pipeline_memory(head)
    assert head.request_id not in budget.reservations
    assert budget.last_decision["envelope"]["prefill_chunk_tokens"] == 8192
    assert not scheduler._prefill_resize_events

    physical["free"] = 16 * GIB
    assert scheduler._reserve_pipeline_memory(head)
    assert budget.reservations[head.request_id].prefill_chunk_tokens == 8192


def test_waiting_does_not_restore_prefix_or_construct_generator(whole_scheduler):
    scheduler, physical = whole_scheduler
    physical["free"] = 2 * GIB
    head = request(scheduler)
    scheduler.waiting.append(head)
    scheduler._ensure_batch_generator = MagicMock(side_effect=AssertionError("model work before complete reservation"))
    scheduler._prepare_prefix_cache_for_request = MagicMock(side_effect=AssertionError("prefix work before complete reservation"))
    scheduled, rejected = scheduler._schedule_waiting()
    assert scheduled == rejected == []
    assert list(scheduler.waiting) == [head]
    assert head.memory_wait_reason == "context_budget"
    scheduler._ensure_batch_generator.assert_not_called()
    scheduler._prepare_prefix_cache_for_request.assert_not_called()


@pytest.mark.parametrize("tokens", [2, 17, 513, 8193, 200000, 371146])
def test_complete_input_is_priced_at_fixed_width_and_output_limit_is_ignored(whole_scheduler, tokens):
    scheduler, _ = whole_scheduler
    short_output = request(scheduler, "short-output", tokens=tokens, maximum=1)
    long_output = request(scheduler, "long-output", tokens=tokens, maximum=512000)
    envelope = scheduler._pipeline_progress_envelope(short_output)
    assert envelope == scheduler._pipeline_progress_envelope(long_output)
    assert envelope.context_tokens == tokens
    assert envelope.prefill_chunk_tokens == min(8192, tokens - 1)
    expected = scheduler._predicted_chunk_transient(
        envelope.prefill_chunk_tokens, tokens - envelope.prefill_chunk_tokens,
        include_shared_reclaim=False,
    )
    assert abs(envelope.prefill_bytes - expected) <= 1
    assert scheduler._admission_charge_tokens(tokens - 1) == min(8192, tokens - 1)


def test_whole_geometry_receives_full_input_horizon_but_never_prices_decode(whole_scheduler):
    scheduler, _ = whole_scheduler
    head = request(scheduler, tokens=198241)
    bound = MagicMock(return_value=GIB)
    scheduler.memory_monitor.estimate_whole_prefill_transient_bytes = bound
    envelope = scheduler._pipeline_progress_envelope(head)
    bound.assert_called_once_with(8192, 198241)
    assert envelope.context_tokens == 198241
    assert envelope.prefill_bytes > GIB

    bound.reset_mock()
    head.output_token_ids.extend([5] * 17)
    decoded = scheduler._pipeline_progress_envelope(head, decoding=True)
    bound.assert_not_called()
    assert decoded.context_tokens == 198241 + 17
    assert decoded.prefill_bytes == 0


@pytest.mark.parametrize("width", [8192, 2048, 17, 1])
def test_compute_tail_boundary_or_fairness_slice_does_not_resize_held_budget(whole_scheduler, width):
    scheduler, _ = whole_scheduler
    head = request(scheduler)
    assert scheduler._reserve_pipeline_memory(head)
    budget = scheduler._request_memory_budget
    envelope = budget.reservations[head.request_id]
    held = budget.reserved_bytes()
    budget.try_resize = MagicMock(side_effect=AssertionError("whole prefill must retain its reservation"))
    assert scheduler._adaptive_chunk_size(width, request_id=head.request_id, loop_label="external", kv_len=163840) == width
    assert scheduler._guard_prefill_chunk(width, kv_len=163840, progress=163840,
                                         loop_label="external", request_id=head.request_id) == width
    assert budget.reservations[head.request_id] is envelope
    assert budget.reserved_bytes() == held
    budget.try_resize.assert_not_called()


def test_optional_upgrade_does_not_change_a_whole_prefill_reservation(whole_scheduler):
    scheduler, _ = whole_scheduler
    head = request(scheduler)
    assert scheduler._reserve_pipeline_memory(head)
    budget = scheduler._request_memory_budget
    # Defensive replay of an old narrower plan: switching policy must not
    # activate the former progress-driven tier-up mechanism.
    previous = replace(budget.reservations[head.request_id], prefill_chunk_tokens=1024)
    budget.reservations[head.request_id] = previous
    budget.try_resize = MagicMock(side_effect=AssertionError("unexpected tier-up"))
    scheduler._maybe_upgrade_prefill_budget(head.request_id, 8192, progress=65536)
    assert budget.reservations[head.request_id] is previous
    budget.try_resize.assert_not_called()


def test_old_decode_and_shared_workspace_remain_charged_at_admission(whole_scheduler):
    scheduler, physical = whole_scheduler
    physical["current"] = 230 * GIB
    old = request(scheduler, "old-decode")
    scheduler.running[old.request_id] = old
    budget = scheduler._request_memory_budget
    budget.reservations[old.request_id] = RequestMemoryEnvelope(200000, 0, 0, 2 * GIB, 2 * GIB)
    budget.mark_decoding(old.request_id)
    scheduler._refresh_pipeline_shared_budget = lambda: None
    budget.shared_reclaim_bytes = GIB
    head = request(scheduler, "new-prefill")

    assert not scheduler._reserve_pipeline_memory(head)
    decision = budget.last_decision
    assert decision["reserved_bytes"] == 2 * GIB
    assert decision["required_bytes"] == decision["envelope"]["total_bytes"] + GIB
    assert list(scheduler.running) == [old.request_id]
    assert head.request_id not in budget.reservations

    budget.release(old.request_id)
    scheduler.running.clear()
    assert scheduler._reserve_pipeline_memory(head)
    assert budget.reservations[head.request_id].prefill_chunk_tokens == 8192


def test_real_pressure_waits_for_the_full_slice_instead_of_shrinking(whole_scheduler):
    scheduler, physical = whole_scheduler
    head = request(scheduler)
    assert scheduler._reserve_pipeline_memory(head)
    budget = scheduler._request_memory_budget
    envelope = budget.reservations[head.request_id]
    physical["current"] = 236 * GIB
    state = SimpleNamespace(request=head, cache=[object()], tokens_processed=163840)
    scheduler._prefill_states[head.request_id] = state
    scheduler._largest_fitting_prefill_chunk = MagicMock(side_effect=AssertionError("must wait, not shrink"))
    assert physical["current"] + scheduler._admission_transient_bound(256, 163840) < 238 * GIB

    with pytest.raises(_PrefillBudgetWait):
        scheduler._guard_prefill_chunk(8192, kv_len=163840, progress=163840,
                                      loop_label="chunked_step", request_id=head.request_id)
    failure = scheduler._last_prefill_budget_failure
    assert failure["chunk_tokens"] == 8192
    assert failure["transient_bytes"] == scheduler._admission_transient_bound(8192, 163840)
    assert failure["stage"] == "waiting_release"
    assert budget.reservations[head.request_id] is envelope
    assert scheduler._prefill_states[head.request_id] is state
    assert state.tokens_processed == 163840
    scheduler._largest_fitting_prefill_chunk.assert_not_called()

    physical["current"] = 220 * GIB
    assert scheduler._guard_prefill_chunk(8192, kv_len=163840, progress=163840,
                                         loop_label="chunked_step", request_id=head.request_id) == 8192
    assert head._prefill_budget_wait_started is None
    assert budget.reservations[head.request_id] is envelope


def test_actual_system_ram_can_still_pause_an_already_reserved_request(whole_scheduler):
    scheduler, physical = whole_scheduler
    head = request(scheduler)
    assert scheduler._reserve_pipeline_memory(head)
    physical["free"] = GIB
    with pytest.raises(_PrefillBudgetWait):
        scheduler._guard_prefill_chunk(8192, kv_len=163840, progress=163840,
                                      loop_label="external", request_id=head.request_id)
    assert scheduler._last_prefill_budget_failure["limit_bytes"] == physical["current"] + GIB
    assert scheduler._last_prefill_budget_failure["chunk_tokens"] == 8192
    physical["free"] = 16 * GIB
    assert scheduler._guard_prefill_chunk(8192, kv_len=163840, progress=163840,
                                         loop_label="external", request_id=head.request_id) == 8192


def test_recovery_clears_wait_text_before_next_progress_without_resetting_speed(whole_scheduler, monkeypatch):
    scheduler, physical = whole_scheduler
    head = request(scheduler)
    assert scheduler._reserve_pipeline_memory(head)
    now = {"value": 100.0}
    monkeypatch.setattr(scheduler_module.time, "monotonic", lambda: now["value"])
    tracker = get_prefill_tracker()
    tracker.update(head.request_id, 8192, head.num_prompt_tokens, "model")
    now["value"] = 102.0
    tracker.update(head.request_id, 16384, head.num_prompt_tokens, "model")
    physical["free"] = GIB
    with pytest.raises(_PrefillBudgetWait):
        scheduler._guard_prefill_chunk(8192, kv_len=16384, progress=16384,
                                      loop_label="external", request_id=head.request_id)
    waiting = tracker.get(head.request_id)
    assert waiting["detail"] == "Waiting for physical memory; progress preserved"
    assert waiting["speed"] > 0

    now["value"] = 103.0
    physical["free"] = 16 * GIB
    assert scheduler._guard_prefill_chunk(8192, kv_len=16384, progress=16384,
                                         loop_label="external", request_id=head.request_id) == 8192
    recovered = tracker.get(head.request_id)
    assert recovered["detail"] is None
    for key in ("start_time", "last_time", "speed", "processed", "total"):
        assert recovered[key] == waiting[key]
    assert scheduler._memory_budget_snapshot()["prefill_guard"]["waiting_request_ids"] == []


def test_materialized_kv_credit_and_decode_handoff_release_only_owned_bytes(whole_scheduler):
    scheduler, physical = whole_scheduler
    head = request(scheduler)
    assert scheduler._reserve_pipeline_memory(head)
    budget = scheduler._request_memory_budget
    envelope = budget.reservations[head.request_id]
    charge = physical["current"] + budget.reserved_bytes()
    materialized = envelope.credit_cap_bytes
    budget.note_materialized(head.request_id, {12345: materialized})
    physical["current"] += materialized
    assert physical["current"] + budget.reserved_bytes() == charge
    budget.mark_decoding(head.request_id)
    assert budget.reserved_bytes() == envelope.decode_bytes - materialized
    assert budget.reservations[head.request_id] is envelope

    scheduler._clear_request_admission_bookkeeping(head.request_id)
    assert head.request_id in budget.reservations  # cleanup still owns the request
    scheduler.requests.pop(head.request_id)
    scheduler._clear_request_admission_bookkeeping(head.request_id)
    assert head.request_id not in budget.reservations
    assert head.request_id not in budget.decoding

    # Reusing the ID starts a fresh whole-prefill reservation without credits.
    scheduler.requests[head.request_id] = head
    assert scheduler._reserve_pipeline_memory(head)
    assert budget.reserved_bytes() == envelope.total_bytes


def test_requeued_waiter_releases_whole_reservation_before_new_admission(whole_scheduler):
    scheduler, _ = whole_scheduler
    head = request(scheduler)
    assert scheduler._reserve_pipeline_memory(head)
    budget = scheduler._request_memory_budget
    envelope = budget.reservations[head.request_id]
    budget.note_materialized(head.request_id, {12345: envelope.credit_cap_bytes})
    scheduler.waiting.append(head)
    scheduler._release_waiting_memory_reservations()
    assert head.request_id not in budget.reservations
    assert head.request_id not in budget._materialized
    scheduler.waiting.clear()
    assert scheduler._reserve_pipeline_memory(head)
    assert budget.reserved_bytes() == envelope.total_bytes


@pytest.mark.parametrize("cached_tokens", [0, 190000])
def test_idle_route_defers_memory_decision_to_complete_scheduler_reservation(whole_scheduler, cached_tokens):
    scheduler, physical = whole_scheduler
    scheduler._prefill_memory_guard = True
    physical["free"] = GIB // 4
    scheduler._admission_estimate = MagicMock(side_effect=AssertionError("route must not use the old partial preflight"))
    assert not scheduler.running and not scheduler.prefilling
    assert scheduler.preflight_eviction_request(
        num_prompt_tokens=200000, cached_tokens=cached_tokens, request_id="route",
    ) is None
    assert scheduler.preflight_or_raise(
        num_prompt_tokens=200000, cached_tokens=cached_tokens, request_id="route",
    ) is None
    scheduler._admission_estimate.assert_not_called()
    assert scheduler._request_memory_budget.reservations == {}


@pytest.mark.asyncio
async def test_idle_engine_route_does_not_evict_or_reject_before_queueing(whole_scheduler):
    from omlx.engine.base import _run_scheduler_preflight_with_cleanup_retry

    scheduler, physical = whole_scheduler
    scheduler._prefill_memory_guard = True
    physical["free"] = GIB // 4
    scheduler._admission_estimate = MagicMock(side_effect=AssertionError("old preflight must not reject the idle route"))
    eviction = AsyncMock(side_effect=AssertionError("the route must not evict to satisfy a deferred reservation"))
    await _run_scheduler_preflight_with_cleanup_retry(
        scheduler, num_prompt_tokens=200000, request_id="idle-route",
        eviction_callback=eviction,
    )
    eviction.assert_not_awaited()
    scheduler._admission_estimate.assert_not_called()


def test_new_idle_request_enters_waiting_instead_of_raising_early_400(whole_scheduler):
    scheduler, physical = whole_scheduler
    scheduler._prefill_memory_guard = True
    physical["free"] = GIB // 4
    head = Request(request_id="idle-queued", prompt=[1] * 8193,
                   sampling_params=SamplingParams(max_tokens=256000))
    scheduler._ensure_batch_generator = MagicMock(side_effect=AssertionError("allocation before complete budget"))
    scheduler._prepare_prefix_cache_for_request = MagicMock(side_effect=AssertionError("restoration before complete budget"))
    try:
        scheduler.add_request(head)
        assert list(scheduler.waiting) == [head]
        assert scheduler.requests[head.request_id] is head
        scheduled, rejected = scheduler._schedule_waiting()
        assert scheduled == rejected == []
        assert list(scheduler.waiting) == [head]
        assert head.memory_wait_reason == "context_budget"
        assert scheduler._request_memory_budget.last_decision["envelope"]["prefill_chunk_tokens"] == 8192
        scheduler._ensure_batch_generator.assert_not_called()
        scheduler._prepare_prefix_cache_for_request.assert_not_called()
    finally:
        scheduler.waiting.clear()
        scheduler.requests.pop(head.request_id, None)
        scheduler._clear_request_admission_bookkeeping(head.request_id)


def test_owned_whole_reservation_is_not_rejected_again_after_prefix_materializes(whole_scheduler):
    scheduler, physical = whole_scheduler
    scheduler._prefill_memory_guard = True
    head = request(scheduler)
    assert scheduler._reserve_pipeline_memory(head)
    held = scheduler._request_memory_budget.reservations[head.request_id]
    # The executor has already obtained the complete budget, then restored a
    # prefix. Reapplying current + full-prompt KV here would double-charge it.
    head.prompt_cache = [object()]
    head.cached_tokens = 190000
    physical["current"] = 225 * GIB
    physical["free"] = GIB
    scheduler._admission_estimate = MagicMock(side_effect=AssertionError("reserved prefill must not use the legacy admission check"))
    assert scheduler._preflight_memory_check(head) is None
    assert scheduler._request_memory_budget.reservations[head.request_id] is held
    scheduler._admission_estimate.assert_not_called()
    # Skipping duplicate admission is not permission to cross the real cap.
    with pytest.raises(_PrefillBudgetWait):
        scheduler._guard_prefill_chunk(8192, kv_len=190000, progress=0,
                                      loop_label="external", request_id=head.request_id)


def test_executor_preflight_is_not_bypassed_without_an_owned_reservation(whole_scheduler):
    scheduler, _ = whole_scheduler
    scheduler._prefill_memory_guard = True
    head = request(scheduler)
    scheduler._admission_estimate = MagicMock(return_value=None)
    assert scheduler._preflight_memory_check(head) is None
    scheduler._admission_estimate.assert_called_once()


def test_rejection_explains_parallel_limits_instead_of_inventing_a_safety_deduction(whole_scheduler):
    scheduler, _ = whole_scheduler
    current, effective = int(220.16 * GIB), int(220.23 * GIB)
    scheduler._last_pipeline_system_budget = {
        "system_limit_bytes": effective,
        "system_free_bytes": effective - current,
        "system_free_reserve_bytes": 0,
    }
    scheduler._last_prefill_peak_caps = {
        "headroom_target_bytes": int(238.8 * GIB),
        "abort_target_bytes": int(239.8 * GIB),
    }
    message = scheduler._format_rejection_message(
        estimated=int(221.66 * GIB), current=current,
        peak=int(1.5 * GIB), hard_limit=effective,
    )
    assert "minimum of applicable ceilings" in message
    assert "effective prefill limit" in message
    assert "process ceiling" in message
    assert "pipeline admission ceiling" in message
    assert "system RAM ceiling" in message
    assert "immediately free RAM" in message
    assert "fixed system reserve" in message
    assert "safety margin" not in message
    assert "19.77" not in message
    assert " - " not in message
