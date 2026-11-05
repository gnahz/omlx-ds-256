"""Bounded scheduling for models whose prefill chunk geometry is numerical state.

No model forward is evaluated. Existing scheduler doubles run on CPU in the
test runner; the policy clock is deterministic rather than wall-clock sleeps.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import mlx.core as mx
import mlx.nn as nn
import pytest

from omlx.decode_activity import get_decode_activity
from omlx.prefill_progress import get_prefill_tracker
from omlx.scheduler import SchedulerConfig
from test_prefill_fairness_budget import (
    add_prefill,
    make_request,
    make_scheduler,
    prefill_ids,
)


@pytest.fixture(autouse=True)
def isolated_activity():
    get_decode_activity().clear()
    get_prefill_tracker().clear()
    with patch("omlx.scheduler._sync_and_clear_cache"):
        yield
    get_decode_activity().clear()
    get_prefill_tracker().clear()


@pytest.fixture
def clock(monkeypatch):
    current = SimpleNamespace(now=100.0)
    monkeypatch.setattr("omlx.scheduler.time.perf_counter", lambda: current.now)
    return current


def preserved_scheduler(**settings):
    scheduler = make_scheduler(prefill_step_size=8192, **settings)
    scheduler.model._omlx_preserve_prefill_chunks = True
    return scheduler


def long_request(request_id="waiting"):
    return make_request(request_id, 20000)


def must_defer(scheduler, request=None, tokens=8192):
    if request is None:
        request = scheduler.waiting[0]
    return scheduler._preserved_prefill_must_defer(request, tokens)


def add_long_prefill(scheduler, request_id):
    request = add_prefill(scheduler, request_id)
    state = scheduler._prefill_states[request_id]
    state.tokens_remaining = mx.zeros((1, 20000), dtype=mx.int32)
    return request


def add_decode(scheduler, request_id="decode"):
    scheduler.running[request_id] = make_request(request_id)


def test_default_is_eight_and_only_real_model_marker_enables_policy(clock):
    assert SchedulerConfig().preserved_prefill_max_defer_seconds == 8.0
    scheduler = make_scheduler(prefill_step_size=8192)
    add_decode(scheduler)
    scheduler.add_request(long_request("waiting"))
    assert not scheduler._preserves_prefill_chunks()  # auto-created MagicMock attrs
    assert not must_defer(scheduler)
    for attribute in ("_language_model", "language_model"):
        scheduler.model = SimpleNamespace(
            **{attribute: SimpleNamespace(_omlx_preserve_prefill_chunks=True)}
        )
        assert scheduler._preserves_prefill_chunks()
    scheduler.model = SimpleNamespace(_omlx_preserve_prefill_chunks=1)
    assert not scheduler._preserves_prefill_chunks()


@pytest.mark.parametrize("wrapper", [None, "_language_model", "language_model"])
def test_mlx_module_capability_and_wrapper_properties_are_read(clock, wrapper):
    class PreservedModule(nn.Module):
        @property
        def _omlx_preserve_prefill_chunks(self):
            return True

    model = PreservedModule()
    assert isinstance(model, dict)
    assert model.get("_omlx_preserve_prefill_chunks") is None
    if wrapper is not None:
        # The wrapper itself is also a descriptor absent from dictionary keys.
        Wrapped = type("Wrapped", (nn.Module,), {
            wrapper: property(lambda self: self["inner"])
        })
        wrapped = Wrapped()
        wrapped["inner"] = model
        assert wrapped.get(wrapper) is None
        model = wrapped
    scheduler = preserved_scheduler()
    scheduler.model = model
    add_decode(scheduler)
    scheduler.add_request(long_request())
    assert scheduler._preserves_prefill_chunks()
    assert scheduler._contended_prefill_cap() == 0
    assert scheduler._prefill_step_size_for_progress(0, 20000) == 8192
    assert must_defer(scheduler)
    clock.now = 108
    assert not must_defer(scheduler)


def test_real_ds41_language_property_keeps_geometry_and_priority(clock):
    from omlx.patches.deepseek_v41.language import LanguageModel
    from test_deepseek_v41 import tiny_ced

    scheduler = preserved_scheduler()
    scheduler.model = LanguageModel(tiny_ced(ced_prefill=True))
    add_decode(scheduler)
    scheduler.add_request(long_request())
    assert scheduler.model._omlx_preserve_prefill_chunks is True
    assert scheduler._preserves_prefill_chunks()
    assert scheduler._contended_prefill_cap() == 0
    assert scheduler._prefill_step_size_for_progress(0, 20000) == 8192
    assert must_defer(scheduler)
    scheduler.config.preserved_prefill_max_defer_seconds = 0
    assert not must_defer(scheduler)
    assert scheduler._prefill_step_size_for_progress(0, 20000) == 8192


def test_plain_dict_explicit_bool_capability_still_works(clock):
    scheduler = preserved_scheduler()
    scheduler.model = {"language_model": {"_omlx_preserve_prefill_chunks": True}}
    assert scheduler._preserves_prefill_chunks()


@pytest.mark.parametrize("value", [-1, float("inf"), float("nan"), True, "8", None])
def test_invalid_defer_config_is_rejected(value):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        SchedulerConfig(preserved_prefill_max_defer_seconds=value)


def test_waiting_prefill_drains_short_decode_without_changing_chunks(clock):
    scheduler = preserved_scheduler()
    add_decode(scheduler)
    waiting = long_request("waiting")
    scheduler.add_request(waiting)
    assert must_defer(scheduler)
    assert scheduler._prefill_step_size_for_progress(0, 100000) == 8192
    clock.now = 105.5
    scheduled, rejected = scheduler._schedule_waiting()
    assert scheduled == rejected == []
    assert list(scheduler.waiting) == [waiting]
    scheduler.running.clear()
    assert not must_defer(scheduler)
    assert scheduler._preserved_prefill_wait_started_at is None
    assert scheduler._prefill_step_size_for_progress(0, 100000) == 8192


def test_live_zero_restores_existing_debt_gate_and_eight_restarts_wait(clock):
    scheduler = preserved_scheduler()
    add_decode(scheduler)
    scheduler.add_request(long_request("waiting"))
    assert must_defer(scheduler)
    clock.now = 102
    scheduler.config.preserved_prefill_max_defer_seconds = 0
    assert not must_defer(scheduler)
    assert scheduler._preserved_prefill_wait_started_at is None
    scheduler._decode_time_owed_s = .5
    assert not scheduler._prefill_gate_open()
    scheduler._repay_decode_debt(.5)
    assert not must_defer(scheduler)
    scheduler.config.preserved_prefill_max_defer_seconds = 8
    assert must_defer(scheduler)
    assert scheduler._preserved_prefill_wait_started_at == 102


def test_new_decodes_and_prefills_do_not_refresh_existing_deadline(clock):
    scheduler = preserved_scheduler()
    add_decode(scheduler)
    scheduler.add_request(long_request("old-waiting"))
    assert must_defer(scheduler)
    for turn in range(1, 8):
        clock.now = 100+turn
        add_decode(scheduler, f"new-decode-{turn}")
        scheduler.add_request(long_request(f"new-waiting-{turn}"))
        assert must_defer(scheduler)
        assert scheduler._preserved_prefill_wait_started_at == 100
    clock.now = 108
    assert not must_defer(scheduler)


def test_expired_deadline_advances_one_full_chunk_then_rotates_fairly(clock):
    scheduler = preserved_scheduler()
    add_decode(scheduler)
    ids = ["prefill-a", "prefill-b", "prefill-c"]
    for request_id in ids:
        add_long_prefill(scheduler, request_id)
    forwarded = []

    def advance(state):
        forwarded.append(state.request.request_id)
        assert scheduler._prefill_step_size_for_progress(0, 100000) == 8192
        clock.now += 4
        scheduler._accrue_decode_debt(4, request_id=state.request.request_id)
        return False

    with patch.object(scheduler, "_step_prefill_chunk", side_effect=advance):
        scheduler._advance_chunked_prefills([], [])
        assert forwarded == []
        for turn, request_id in enumerate(ids*2):
            clock.now += 8
            scheduler._repay_decode_debt(scheduler._decode_time_owed_s)
            scheduler._advance_chunked_prefills([], [])
            assert forwarded == (ids*2)[:turn+1]
            assert prefill_ids(scheduler)[-1] == request_id
            scheduler._advance_chunked_prefills([], [])
            assert len(forwarded) == turn+1
    assert prefill_ids(scheduler) == ids


def test_disabled_fairness_and_no_decode_remain_immediate(clock):
    scheduler = preserved_scheduler(decode_fairness=False)
    add_decode(scheduler)
    scheduler.add_request(long_request("waiting"))
    assert not must_defer(scheduler)
    scheduler._decode_fairness = True
    scheduler.running.clear()
    assert not must_defer(scheduler)
    assert scheduler._preserved_prefill_wait_started_at is None


def test_own_debt_and_other_engine_hold_still_apply_after_deadline(clock):
    scheduler = preserved_scheduler()
    add_decode(scheduler)
    scheduler.add_request(long_request("waiting"))
    assert must_defer(scheduler)
    clock.now = 108
    scheduler._decode_time_owed_s = 1
    assert not scheduler._prefill_gate_open()
    scheduler._repay_decode_debt(1)
    get_decode_activity().extend_hold(110)
    assert not scheduler._prefill_gate_open()
    clock.now = 110
    assert not must_defer(scheduler)


def test_other_engine_decode_gets_bounded_priority_too(clock):
    scheduler = preserved_scheduler()
    scheduler.add_request(long_request("waiting"))
    get_decode_activity().publish("other-decoder", 1)
    assert must_defer(scheduler)
    clock.now = 108
    assert not must_defer(scheduler)
    get_decode_activity().publish("other-decoder", 0)
    assert not must_defer(scheduler)
    assert scheduler._preserved_prefill_wait_started_at is None


def test_cancelled_wait_does_not_expire_new_request_priority(clock):
    scheduler = preserved_scheduler()
    add_decode(scheduler)
    scheduler.add_request(long_request("cancelled"))
    assert must_defer(scheduler)
    scheduler.waiting.clear()  # cancellation can replace queue before another step
    clock.now = 120
    scheduler.add_request(long_request("new-waiting"))
    assert must_defer(scheduler)
    assert scheduler._preserved_prefill_wait_started_at == 120
    scheduler.waiting.clear()
    assert scheduler._prefill_gate_open()
    assert scheduler._preserved_prefill_wait_request_ids == set()


def test_step_services_decode_while_both_prefill_entrypoints_are_held(clock):
    scheduler = preserved_scheduler()
    add_decode(scheduler)
    add_long_prefill(scheduler, "inflight")
    scheduler.add_request(long_request("new-waiting"))
    scheduler.batch_generator.next_generated.return_value = iter([])
    with (
        patch.object(scheduler, "_step_prefill_chunk") as advance,
        patch.object(scheduler, "_begin_prefill") as begin,
        patch.object(scheduler, "_do_external_prefill") as external,
    ):
        result = scheduler.step()
    advance.assert_not_called()
    begin.assert_not_called()
    external.assert_not_called()
    scheduler.batch_generator.next_generated.assert_called_once()
    assert result.has_work


def test_reset_clears_pending_decode_priority(clock):
    scheduler = preserved_scheduler()
    add_decode(scheduler)
    scheduler.add_request(long_request("waiting"))
    assert must_defer(scheduler)
    scheduler.reset()
    assert scheduler._preserved_prefill_wait_started_at is None
    assert scheduler._preserved_prefill_wait_request_ids == set()


def test_prepared_long_request_is_restored_once_while_waiting(clock):
    scheduler = preserved_scheduler()
    add_decode(scheduler)
    request = make_request("partial-long", 30000)
    scheduler.add_request(request)
    cache = []
    exact = SimpleNamespace(prepare=MagicMock(return_value=(cache, 8192, object())))
    with (
        patch.object(scheduler, "_configure_exact_prefix_cache", return_value=exact),
        patch.object(scheduler, "_validate_cache", return_value=True),
    ):
        for now in (100, 101, 107.999):
            clock.now = now
            scheduled, rejected = scheduler._schedule_waiting()
            assert scheduled == rejected == []
            assert list(scheduler.waiting) == [request]
            assert request.prompt_cache is cache
            assert request.remaining_tokens == request.prompt_token_ids[8192:]
    exact.prepare.assert_called_once_with(request.prompt_token_ids, 8192)
    assert scheduler._preserved_prefill_wait_started_at == 100


@pytest.mark.parametrize("cached,total", [(8192, 8242), (0, 257)])
def test_hot_cache_tail_and_partial_short_prompt_do_not_wait(clock, cached, total):
    scheduler = preserved_scheduler()
    add_decode(scheduler)
    request = make_request("short", total)
    scheduler.add_request(request)
    exact = SimpleNamespace(prepare=MagicMock(return_value=([], cached, object())))
    with (
        patch.object(scheduler, "_configure_exact_prefix_cache", return_value=exact),
        patch.object(scheduler, "_validate_cache", return_value=True),
        patch.object(scheduler, "_do_external_prefill", return_value=([], [total-1])) as forward,
    ):
        scheduled, rejected = scheduler._schedule_waiting()
    assert scheduled == [request]
    assert rejected == []
    assert len(forward.call_args.args[1]) == total-cached
    assert scheduler._preserved_prefill_wait_started_at is None


def test_long_head_does_not_block_eight_hot_tails_or_lose_its_deadline(clock):
    scheduler = preserved_scheduler(max_num_seqs=16)
    add_decode(scheduler)
    long = long_request("long")
    scheduler.add_request(long)
    short = [make_request(f"hot-{index}", 8242) for index in range(8)]
    for request in short:
        scheduler.add_request(request)
    exact = SimpleNamespace(
        prepare=MagicMock(side_effect=lambda tokens, step: (
            [], 8192 if len(tokens) == 8242 else 0, object()
        ))
    )

    def forward(request, tokens, cache, **kwargs):
        scheduler._accrue_decode_debt(.02, request_id=request.request_id)
        return [], tokens[-1:]

    with (
        patch.object(scheduler, "_configure_exact_prefix_cache", return_value=exact),
        patch.object(scheduler, "_validate_cache", return_value=True),
        patch.object(scheduler, "_do_external_prefill", side_effect=forward),
    ):
        admitted = []
        for now in (100, 101, 102, 103, 104, 105, 106, 107):
            clock.now = now
            scheduler._repay_decode_debt(scheduler._decode_time_owed_s)
            scheduled, rejected = scheduler._schedule_waiting()
            admitted.extend(scheduled)
            assert rejected == []
            assert scheduler._preserved_prefill_wait_started_at == 100
        assert admitted == short
        assert list(scheduler.waiting) == [long]
        assert exact.prepare.call_count == 9
        clock.now = 108
        scheduler._repay_decode_debt(scheduler._decode_time_owed_s)
        assert not must_defer(scheduler)


def test_short_inflight_tail_bypasses_long_chunk_without_refreshing_timer(clock):
    scheduler = preserved_scheduler()
    add_decode(scheduler)
    long = add_long_prefill(scheduler, "long")
    tail = add_prefill(scheduler, "tail")
    assert must_defer(scheduler, long)
    clock.now = 107
    forwarded = []

    def advance(state):
        forwarded.append(state.request.request_id)
        scheduler._accrue_decode_debt(.01, request_id=state.request.request_id)
        return False

    with patch.object(scheduler, "_step_prefill_chunk", side_effect=advance):
        scheduler._advance_chunked_prefills([], [])
    assert forwarded == [tail.request_id]
    assert scheduler._preserved_prefill_wait_started_at == 100
    clock.now = 108
    scheduler._repay_decode_debt(.01)
    assert not must_defer(scheduler, long)


def test_cancel_prepared_wait_cleans_cache_marker_and_priority(clock):
    scheduler = preserved_scheduler()
    add_decode(scheduler)
    request = long_request("cancelled")
    scheduler.add_request(request)
    exact = SimpleNamespace(prepare=MagicMock(return_value=([], 8192, object())))
    with patch.object(scheduler, "_configure_exact_prefix_cache", return_value=exact):
        scheduler._schedule_waiting()
    assert request.request_id in scheduler._prefix_cache_prepared
    assert scheduler._preserved_prefill_wait_started_at == 100
    scheduler.abort_request(request.request_id)
    scheduler._process_pending_aborts()
    assert not scheduler.waiting
    assert request.request_id not in scheduler._prefix_cache_prepared
    assert scheduler._preserved_prefill_wait_request_ids == set()
    assert scheduler._preserved_prefill_wait_started_at is None


def test_other_engine_hold_sleeps_but_released_chunk_reports_work(clock):
    scheduler = preserved_scheduler()
    add_long_prefill(scheduler, "long")
    get_decode_activity().publish("other-decoder", 1)

    def advance(state):
        scheduler._accrue_decode_debt(4, request_id=state.request.request_id)
        return False

    with patch.object(scheduler, "_step_prefill_chunk", side_effect=advance) as forward:
        result = scheduler.step()
        assert not result.has_work
        forward.assert_not_called()
        clock.now = 108
        result = scheduler.step()
        assert result.has_work
        forward.assert_called_once()


def test_measured_budget_classifies_cost_without_resizing_chunk(clock):
    scheduler = preserved_scheduler()
    add_decode(scheduler)
    request = long_request()
    scheduler.add_request(request)
    scheduler._prefill_tps_best = 2000
    assert scheduler._contended_prefill_token_budget() == 960
    assert not must_defer(scheduler, request, 960)
    assert must_defer(scheduler, request, 961)
    assert scheduler._prefill_step_size_for_progress(0, 20000) == 8192


def test_invalid_short_cache_uses_full_prompt_cost(clock):
    scheduler = preserved_scheduler()
    add_decode(scheduler)
    request = make_request("invalid-hit", 8242)
    scheduler.add_request(request)
    exact = SimpleNamespace(prepare=MagicMock(return_value=([], 8192, object())))
    with (
        patch.object(scheduler, "_configure_exact_prefix_cache", return_value=exact),
        patch.object(scheduler, "_validate_cache", return_value=False),
        patch.object(scheduler, "_do_external_prefill") as forward,
    ):
        scheduled, rejected = scheduler._schedule_waiting()
    assert scheduled == rejected == []
    assert list(scheduler.waiting) == [request]
    assert request.remaining_tokens == request.prompt_token_ids
    assert request.cached_tokens == 0
    forward.assert_not_called()
    assert scheduler._preserved_prefill_wait_started_at == 100


def test_live_zero_admits_original_full_first_chunk_in_same_process(clock):
    scheduler = preserved_scheduler()
    add_decode(scheduler)
    request = long_request()
    scheduler.add_request(request)
    with patch.object(scheduler, "_step_prefill_chunk", return_value=False) as forward:
        scheduler._schedule_waiting()
        forward.assert_not_called()
        scheduler.config.preserved_prefill_max_defer_seconds = 0
        scheduler._schedule_waiting()
    forward.assert_called_once()
    state = forward.call_args.args[0]
    assert state.request is request
    assert state.tokens_remaining.shape[1] == 19999
    assert scheduler._prefill_step_size_for_progress(0, 19999) == 8192
    assert scheduler._preserved_prefill_wait_started_at is None


@pytest.mark.parametrize("chunked", [False, True])
def test_formerly_expensive_request_short_tail_keeps_other_deadline(clock, chunked):
    scheduler = preserved_scheduler()
    add_decode(scheduler)
    b = add_long_prefill(scheduler, "B")
    c = add_long_prefill(scheduler, "C")
    assert must_defer(scheduler, b)
    assert must_defer(scheduler, c)
    clock.now = 112
    scheduler._accrue_decode_debt(4, request_id="B", forward_tokens=8192)
    scheduler._repay_decode_debt(scheduler._decode_time_owed_s)
    clock.now = 119
    assert must_defer(scheduler, c)
    assert not must_defer(scheduler, b, 43)
    assert "B" in scheduler._preserved_prefill_wait_request_ids
    scheduler.model.side_effect = lambda *args, **kwargs: None
    with (
        patch("omlx.scheduler.make_prompt_cache", return_value=[]),
        patch.object(scheduler, "_announce_first_prefill_chunk"),
        patch.object(scheduler, "_reserve_qsa_index_capacity"),
    ):
        if chunked:
            state = scheduler._prefill_states["B"]
            state.tokens_remaining = mx.zeros((1, 43), dtype=mx.int32)
            scheduler._step_prefill_chunk(state)
        else:
            scheduler._do_external_prefill(b, list(range(44)), [])
    assert scheduler._preserved_prefill_wait_started_at == 112
    clock.now = 120
    assert not must_defer(scheduler, c)
