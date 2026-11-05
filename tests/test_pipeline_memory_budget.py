"""No allocations: replay long-context admission with byte counters."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from omlx.request_memory_budget import GIB, RequestMemoryBudget, RequestMemoryEnvelope, estimate_envelope
from omlx.scheduler import Scheduler, SchedulerConfig
import omlx.scheduler as scheduler_module
from omlx.request import Request, SamplingParams


@pytest.mark.parametrize("maximum", [1, 8192, 256000, 512000])
def test_output_limit_never_changes_progress_budget(maximum):
    monitor = SimpleNamespace(estimate_resident_kv_bytes=lambda count, **kw: count * 890)
    e = estimate_envelope(monitor, 437982, maximum, 8192, lambda q, kv: q * kv)
    baseline = estimate_envelope(monitor, 437982, 1, 8192, lambda q, kv: q * kv)
    assert e == baseline
    assert e.context_tokens == 437982
    assert e.prefill_bytes == 8192 * (437982 - 8192)
    assert e.decode_bytes >= e.resident_bytes


def test_four_short_requests_fit_but_long_context_waits_until_release():
    budget = RequestMemoryBudget()
    short = RequestMemoryEnvelope(100, 0, GIB, GIB, GIB)
    long = RequestMemoryEnvelope(950000, 3*GIB, 8*GIB, 3*GIB, 11*GIB)
    for i in range(4):
        assert budget.reserve(str(i), short, 222*GIB, 240*GIB)
    assert budget.reserve("long", long, 222*GIB, 240*GIB)
    assert not budget.reserve("other", long, 222*GIB, 240*GIB)
    assert budget.last_decision["limit_bytes"] == 238*GIB
    for rid in list(budget.reservations):
        budget.release(rid)
    assert budget.reserve("other", long, 222*GIB, 240*GIB)


@pytest.mark.parametrize("ceiling", [0, 230*GIB, 240*GIB, 254*GIB])
def test_dynamic_limit_can_only_tighten_240_gib_policy(ceiling):
    budget = RequestMemoryBudget()
    assert budget.limit(ceiling) == min(ceiling, 238*GIB)
    e = RequestMemoryEnvelope(10, 0, GIB, GIB, GIB)
    assert not budget.reserve("too-big", e, 238*GIB, ceiling)


@pytest.fixture(autouse=True)
def system_telemetry(monkeypatch):
    monkeypatch.setattr(scheduler_module, "get_phys_footprint", lambda: 232*GIB)
    monkeypatch.setattr(scheduler_module, "get_macos_vm_stats", lambda: {"free": 64*GIB})


@pytest.fixture
def scheduler(monkeypatch):
    monkeypatch.setenv("OMLX_PIPELINE_MEMORY_BUDGET", "1")
    model, tokenizer = MagicMock(), MagicMock()
    model.layers = []
    model.config = SimpleNamespace(num_hidden_layers=40, num_key_value_heads=1, num_attention_heads=64, head_dim=512)
    del model.make_cache
    tokenizer.eos_token_id = 2
    s = Scheduler(model, tokenizer, SchedulerConfig(max_num_seqs=4, paged_cache_block_size=0, prefill_step_size=8192, prefill_peak_target_gib=238))
    s._memory_hard_limit_bytes = 240*GIB
    s._memory_abort_limit_bytes = 240*GIB
    s._memory_hard_watermark_bytes = int(239.8*GIB)
    s._prefill_headroom_safety = s._prefill_abort_margin = .995
    s._current_usage_bytes = lambda **kw: 232*GIB
    s._predicted_chunk_transient = lambda query, context, **kwargs: min(query, 8192) * 1024**2
    s.memory_monitor.estimate_resident_kv_bytes = lambda count, **kw: count * 890
    s._prefill_gate_open = lambda: True
    s._should_defer_for_cache_freshness = lambda req: False
    s._store_cache_gate = None
    s._memory_limit_bytes = 0
    s._admission_paused = False
    return s


def test_system_limit_counts_only_released_ram_and_never_file_cache():
    from omlx.request_memory_budget import physical_headroom_budget
    b = physical_headroom_budget(222*GIB, {"free": 12*GIB, "speculative": 2*GIB,
                                        "inactive": 20*GIB, "external": 30*GIB})
    assert b["system_free_bytes"] == 10*GIB
    assert b["system_limit_bytes"] == 232*GIB
    assert b["system_free_reserve_bytes"] == 0


@pytest.mark.parametrize("stats", [
    {"free": 0},
    {"free": -GIB},
    {"free": GIB, "speculative": 2*GIB},
    {"free": 0, "inactive": 20*GIB, "external": 30*GIB},
])
def test_no_released_ram_cannot_admit_a_positive_allocation(stats):
    from omlx.request_memory_budget import physical_headroom_budget
    current = 222*GIB
    headroom = physical_headroom_budget(current, stats)
    assert headroom["system_free_bytes"] == 0
    assert headroom["system_limit_bytes"] == current
    assert headroom["system_free_reserve_bytes"] == 0
    budget = RequestMemoryBudget()
    one_byte = RequestMemoryEnvelope(1, 0, 1, 1, 1)
    assert not budget.reserve("new", one_byte, current, headroom["system_limit_bytes"])
    assert budget.reservations == {}


@pytest.mark.parametrize("free", [GIB, 16*GIB, 64*GIB])
def test_removing_fixed_reserve_never_raises_process_or_admission_ceiling(free):
    from omlx.request_memory_budget import physical_headroom_budget
    current = 237*GIB
    headroom = physical_headroom_budget(current, {"free": free})
    budget = RequestMemoryBudget()
    full = RequestMemoryEnvelope(10, 0, GIB, GIB, GIB)
    assert budget.reserve("fits", full, current, headroom["system_limit_bytes"])
    assert budget.last_decision["limit_bytes"] == 238*GIB
    assert budget.snapshot(current, headroom["system_limit_bytes"])["process_limit_bytes"] == 240*GIB
    assert not budget.reserve("one-more", RequestMemoryEnvelope(1, 0, 1, 1, 1),
                              current, headroom["system_limit_bytes"])


def test_system_limit_fails_closed_without_telemetry():
    from omlx.request_memory_budget import physical_headroom_budget
    for footprint, stats in [(222*GIB, None), (0, {"free": 64*GIB}), (222*GIB, {}),
                             (222*GIB, {"free": True}), (222*GIB, {"free": "64"})]:
        assert physical_headroom_budget(footprint, stats)["system_limit_bytes"] == 1


def test_context_stall_advice_does_not_suggest_raising_the_ceiling(scheduler, monkeypatch):
    request = Request(request_id="blocked", prompt=[1], sampling_params=SamplingParams(max_tokens=16))
    scheduler.add_request(request)
    monkeypatch.setattr(scheduler_module.time, "monotonic", lambda: 100.)
    assert scheduler._memory_admission_stall_output("context_budget") is None
    monkeypatch.setattr(scheduler_module.time, "monotonic", lambda: 161.)
    output = scheduler._memory_admission_stall_output("context_budget")
    assert output.error_code == "memory_admission_stalled"
    assert "Free physical RAM" in output.error
    assert "Raise custom_ceiling" not in output.error


def test_new_prefill_waits_for_system_ram_while_old_decode_fits(scheduler, monkeypatch):
    monkeypatch.setattr(scheduler_module, "get_phys_footprint", lambda: 222*GIB)
    monkeypatch.setattr(scheduler_module, "get_macos_vm_stats", lambda: {"free": 4*GIB})
    scheduler._current_usage_bytes = lambda **kw: 222*GIB
    request = Request(request_id="old", prompt=[1], sampling_params=SamplingParams(max_tokens=16))
    scheduler.running["old"] = request
    budget = scheduler._request_memory_budget
    budget.reservations["old"] = RequestMemoryEnvelope(10, GIB, GIB, GIB, 2*GIB)
    budget.mark_decoding("old")
    assert scheduler._pipeline_decode_fits()
    assert scheduler._last_decode_budget_decision["limit_bytes"] == 226*GIB
    new = Request(request_id="new", prompt=[1], sampling_params=SamplingParams(max_tokens=256000))
    new.num_prompt_tokens = 60000
    assert not scheduler._reserve_pipeline_memory(new)
    assert budget.last_decision["limit_bytes"] == 226*GIB
    # Only a measured rise in free memory opens admission.
    monkeypatch.setattr(scheduler_module, "get_macos_vm_stats", lambda: {"free": 24*GIB})
    assert scheduler._reserve_pipeline_memory(new)


def test_system_pressure_tightens_live_decode_gate(scheduler, monkeypatch):
    monkeypatch.setattr(scheduler_module, "get_phys_footprint", lambda: 222*GIB)
    monkeypatch.setattr(scheduler_module, "get_macos_vm_stats", lambda: {"free": 0})
    scheduler._current_usage_bytes = lambda **kw: 222*GIB
    request = Request(request_id="old", prompt=[1], sampling_params=SamplingParams(max_tokens=16))
    scheduler.running["old"] = request
    scheduler._request_memory_budget.reservations["old"] = RequestMemoryEnvelope(10, GIB, GIB, GIB, 2*GIB)
    assert not scheduler._pipeline_decode_fits()
    assert scheduler._last_decode_budget_decision["limit_bytes"] == 222*GIB
    assert scheduler._last_decode_budget_decision["system_free_bytes"] == 0


def test_long_request_waits_without_restoring_prefix_or_allocating_batch(scheduler):
    request = Request(request_id="long", prompt=[1], sampling_params=SamplingParams(max_tokens=512000))
    request.num_prompt_tokens = 437982
    scheduler.requests["long"] = request
    scheduler.waiting.append(request)
    scheduler.running["running"] = SimpleNamespace(request_id="running")
    scheduler._ensure_batch_generator = MagicMock(side_effect=AssertionError("allocated before budget"))
    scheduler._prepare_prefix_cache_for_request = MagicMock(side_effect=AssertionError("restored before budget"))
    scheduled, rejected = scheduler._schedule_waiting()
    assert scheduled == rejected == []
    assert list(scheduler.waiting) == [request]
    assert request.memory_wait_reason == "context_budget"
    scheduler._ensure_batch_generator.assert_not_called()
    scheduler._prepare_prefix_cache_for_request.assert_not_called()


def test_decode_checks_headroom_again_before_model_call(scheduler):
    request = Request(request_id="one", prompt=[1], sampling_params=SamplingParams(max_tokens=96))
    request.num_prompt_tokens = 38
    scheduler._current_usage_bytes = lambda **kw: 222*GIB
    assert scheduler._reserve_pipeline_memory(request)
    scheduler.running["one"] = request
    assert scheduler._pipeline_decode_fits()
    scheduler._current_usage_bytes = lambda **kw: 238*GIB
    assert not scheduler._pipeline_decode_fits()


def test_cleanup_releases_reservation_only_after_request_removed(scheduler):
    request = Request(request_id="one", prompt=[1], sampling_params=SamplingParams())
    scheduler._current_usage_bytes = lambda **kw: 220*GIB
    assert scheduler._reserve_pipeline_memory(request)
    scheduler.requests["one"] = request
    scheduler._clear_request_admission_bookkeeping("one")
    assert "one" in scheduler._request_memory_budget.reservations
    scheduler.requests.pop("one")
    scheduler._clear_request_admission_bookkeeping("one")
    assert not scheduler._request_memory_budget.reservations


def test_physical_usage_never_credits_pending_release_or_hot_cache(scheduler, monkeypatch):
    monkeypatch.setattr(scheduler_module, "get_phys_footprint", lambda: 236*GIB)
    monkeypatch.setattr(scheduler_module, "unreleased_graphics_bytes", lambda *a, **kw: 8*GIB)
    monkeypatch.setattr(scheduler_module.mx, "get_active_memory", lambda: 220*GIB)
    monkeypatch.setattr(scheduler_module.mx, "get_cache_memory", lambda: 4*GIB)
    scheduler._hot_cache_cpu_bytes = lambda: 2*GIB
    assert Scheduler._current_usage_bytes(scheduler) == 236*GIB


def test_budget_still_waits_while_reclaimed_pages_are_physically_resident(scheduler):
    request = Request(request_id="long", prompt=[1], sampling_params=SamplingParams(max_tokens=512000))
    request.num_prompt_tokens = 437982
    scheduler.requests["long"] = request
    scheduler.waiting.append(request)
    scheduler._reclaim_prefill_headroom = MagicMock()
    scheduler._ensure_batch_generator = MagicMock(side_effect=AssertionError("allocated before physical release"))
    scheduled, rejected = scheduler._schedule_waiting()
    assert scheduled == rejected == []
    assert list(scheduler.waiting) == [request]
    scheduler._reclaim_prefill_headroom.assert_called_once()
    scheduler._ensure_batch_generator.assert_not_called()


def test_decode_transition_releases_prefill_allowance_but_retains_growth_until_cleanup():
    budget = RequestMemoryBudget()
    envelope = RequestMemoryEnvelope(524724, GIB, 6*GIB, 2*GIB, 7*GIB)
    assert budget.reserve("old", envelope, 220*GIB, 240*GIB)
    assert budget.reserved_bytes() == 7*GIB
    budget.mark_decoding("old")
    assert budget.reserved_bytes() == 2*GIB
    assert budget.reserve("next", envelope, 220*GIB, 240*GIB)
    assert budget.decode_additional_bytes(["old"]) == 2*GIB
    budget.release("old")
    assert "old" not in budget.decoding
    assert budget.reserved_bytes() == 7*GIB
    assert budget.reserve("old", envelope, 220*GIB, 240*GIB)
    assert budget.reserved_bytes() == 14*GIB  # requeue reserves prefill again


def test_budget_snapshot_reports_live_shortfall_and_clears_stale_blocker():
    budget = RequestMemoryBudget()
    envelope = RequestMemoryEnvelope(524724, GIB, 6*GIB, 2*GIB, 7*GIB)
    assert budget.reserve("decode", envelope, 220*GIB, 240*GIB)
    budget.mark_decoding("decode")
    assert not budget.reserve("head", envelope, 232*GIB, 240*GIB)
    snap = budget.snapshot(232*GIB, 240*GIB, "head")
    assert snap["reserved_bytes"] == 2*GIB
    assert snap["limit_bytes"] == 238*GIB
    assert snap["process_limit_bytes"] == 240*GIB
    assert snap["pending"]["shortfall_bytes"] == 3*GIB
    assert snap["pending"]["envelope"]["context_tokens"] == 524724
    assert budget.snapshot(220*GIB, 240*GIB)["pending"] is None


def test_waiting_releases_logical_budget_but_cleanup_keeps_it(scheduler):
    request = Request(request_id="wait", prompt=[1], sampling_params=SamplingParams(max_tokens=512000))
    scheduler._current_usage_bytes = lambda **kw: 220*GIB
    assert scheduler._reserve_pipeline_memory(request)
    scheduler.requests["wait"] = request
    scheduler.waiting.append(request)
    scheduler._publish_admin_snapshot()
    assert "wait" not in scheduler._request_memory_budget.reservations
    scheduler.waiting.clear()
    assert scheduler._reserve_pipeline_memory(request)
    scheduler._request_memory_budget.mark_decoding("wait")
    scheduler._pending_async_removes = [(1, "wait", MagicMock())]
    scheduler._publish_admin_snapshot()
    assert "wait" in scheduler._request_memory_budget.reservations
    scheduler.requests.pop("wait")
    scheduler._pipeline_owned_ids = {"wait"}
    scheduler._pending_async_removes = []
    scheduler._publish_admin_snapshot()
    assert "wait" not in scheduler._request_memory_budget.reservations


def test_fifo_head_survives_continuous_new_arrivals_and_decode_priority(monkeypatch):
    from unittest.mock import patch
    from test_prefill_fairness_budget import make_scheduler, make_request
    from omlx.decode_activity import get_decode_activity
    get_decode_activity().clear()
    monkeypatch.setenv("OMLX_PIPELINE_MEMORY_BUDGET", "1")
    clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr(scheduler_module.time, "perf_counter", lambda: clock.now)
    scheduler = make_scheduler(max_num_seqs=4, prefill_step_size=8192, chunked_prefill=False)
    scheduler.model._omlx_preserve_prefill_chunks = True
    scheduler._memory_hard_limit_bytes = 240*GIB
    scheduler._memory_abort_limit_bytes = 240*GIB
    scheduler._memory_hard_watermark_bytes = 240*GIB
    scheduler._prefill_headroom_safety = scheduler._prefill_abort_margin = .995
    scheduler._current_usage_bytes = lambda **kw: 220*GIB
    scheduler._store_cache_gate = None
    scheduler._prefill_memory_guard = False
    envelope = RequestMemoryEnvelope(532000, GIB, 6*GIB, 2*GIB, 7*GIB)
    monkeypatch.setattr(scheduler_module, "estimate_envelope", lambda *args, **kwargs: envelope)
    decoder = make_request("decoder")
    scheduler.requests["decoder"] = decoder
    scheduler.running["decoder"] = decoder
    assert scheduler._reserve_pipeline_memory(decoder)
    scheduler._request_memory_budget.mark_decoding("decoder")
    oldest = make_request("oldest", 20000)
    scheduler.add_request(oldest)
    with (patch.object(Scheduler, "_packed_prefill_ready", return_value=False),
          patch.object(scheduler, "_step_prefill_chunk", return_value=True),
          patch.object(scheduler, "_do_external_prefill", side_effect=lambda req, tokens, cache, **kw: ([], tokens[-1:])),
          patch.object(scheduler, "_preflight_memory_check", return_value=None)):
        for tick in range(8):
            clock.now = 100 + tick
            if tick in (1, 2):
                scheduler.add_request(make_request(f"new-{tick}", 8242))
            scheduled, rejected = scheduler._schedule_waiting()
            assert scheduled == rejected == []
            assert scheduler.waiting[0] is oldest
            assert "oldest" not in scheduler._request_memory_budget.reservations
            assert len(scheduler.requests) <= 4
        clock.now = 108
        scheduled, rejected = scheduler._schedule_waiting()
        assert scheduled[0] is oldest
        assert rejected == []
        assert "oldest" in scheduler.running
        assert "oldest" in scheduler._request_memory_budget.decoding
        assert scheduler._request_memory_budget.reserved_bytes() <= 18*GIB
    get_decode_activity().clear()


def test_empty_budget_snapshot_has_no_blocker():
    snap = RequestMemoryBudget().snapshot(220*GIB, 240*GIB)
    assert snap["pending"] is None
    assert snap["reservations"] == []
    assert snap["reserved_bytes"] == 0


def test_requeue_prices_folded_output_as_prefill_without_extending_output_budget(scheduler, monkeypatch):
    request = Request(request_id="retry", prompt=[1], sampling_params=SamplingParams(max_tokens=512000))
    request.num_prompt_tokens = 10000
    request.requeued_output_tokens = 20000
    scheduler._current_usage_bytes = lambda **kw: 220*GIB
    estimate = MagicMock(return_value=RequestMemoryEnvelope(522000, GIB, GIB, GIB, 2*GIB))
    monkeypatch.setattr(scheduler_module, "estimate_envelope", estimate)
    assert scheduler._reserve_pipeline_memory(request)
    assert estimate.call_args.args[1:3] == (30000, 0)


def test_impossible_head_errors_after_no_progress_timeout_without_leaking_slot(scheduler, monkeypatch):
    clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr(scheduler_module.time, "monotonic", lambda: clock.now)
    request = Request(request_id="impossible", prompt=[1], sampling_params=SamplingParams(max_tokens=512000))
    scheduler.add_request(request)
    request.num_prompt_tokens = 437982
    scheduler._reclaim_prefill_headroom = MagicMock()
    scheduler._MEMORY_ADMISSION_STALL_TIMEOUT_S = 10
    scheduled, errors = scheduler._schedule_waiting()
    assert scheduled == errors == []
    clock.now = 110.0
    scheduled, errors = scheduler._schedule_waiting()
    assert not scheduled
    assert [error.error_code for error in errors] == ["memory_admission_stalled"]
    scheduler._publish_admin_snapshot()
    assert not scheduler.waiting
    assert request.request_id not in scheduler.requests
    assert not scheduler._request_memory_budget.reservations
    assert scheduler.snapshot_for_admin()["memory_budget"]["pending"] is None


def test_cancelled_deferred_head_releases_budget_and_removes_stale_blocker(scheduler):
    scheduler._current_usage_bytes = lambda **kw: 220*GIB
    request = Request(request_id="cancel", prompt=[1], sampling_params=SamplingParams(max_tokens=1))
    scheduler.add_request(request)
    assert scheduler._reserve_pipeline_memory(request)
    scheduler._do_abort_request("cancel", allow_cache_store=False)
    scheduler._publish_admin_snapshot()
    assert not scheduler.waiting
    assert "cancel" not in scheduler._request_memory_budget.reservations
    assert scheduler.snapshot_for_admin()["memory_budget"]["pending"] is None


def test_sync_prefill_publishes_budget_before_blocking_executor(scheduler, monkeypatch):
    scheduler._current_usage_bytes = lambda **kw: 220*GIB
    request = Request(request_id="sync-prefill", prompt=list(range(38)),
                      sampling_params=SamplingParams(max_tokens=512000))
    scheduler.add_request(request)
    scheduler.batch_generator = MagicMock()
    scheduler.batch_generator.insert.return_value = [1]
    scheduler._ensure_batch_generator = MagicMock()
    scheduler._prepare_prefix_cache_for_request = MagicMock()
    scheduler._preflight_memory_check = MagicMock(return_value=None)
    observed = []
    def prefill(req, tokens, cache, **kwargs):
        snap = scheduler.snapshot_for_admin()
        assert snap["waiting"][0] is request
        reservations = snap["memory_budget"]["reservations"]
        assert reservations[0]["request_id"] == "sync-prefill"
        assert reservations[0]["phase"] == "prefill"
        assert reservations[0]["reserved_bytes"] > 0
        observed.append(snap["memory_budget"]["reserved_bytes"])
        return [], tokens[-1:]
    scheduler._do_external_prefill = prefill
    scheduled, errors = scheduler._schedule_waiting()
    assert scheduled == [request] and not errors
    assert observed
    scheduler._publish_admin_snapshot()
    final = scheduler.snapshot_for_admin()["memory_budget"]
    assert final["reservations"][0]["phase"] == "decode"
    assert final["reserved_bytes"] < observed[0]


def test_resident_is_not_added_twice_when_decode_workspace_dominates():
    monitor = SimpleNamespace(estimate_resident_kv_bytes=lambda *a, **kw: GIB)
    e = estimate_envelope(monitor, 10, 100, 8192, lambda q, c: GIB if q == 5 else GIB//2)
    assert e.resident_bytes == 3*GIB
    assert e.decode_bytes == e.total_bytes == 4*GIB


def test_reclaimed_workspace_is_shared_once_and_225_gib_head_can_run():
    budget = RequestMemoryBudget()
    budget.shared_reclaim_bytes = 2*GIB
    # Three existing rows: 1 GiB private allowance + formerly 2 GiB shared
    # overhead EACH. A fourth prefill needs 6 GiB private plus that same pool.
    e = RequestMemoryEnvelope(317000, GIB, 0, GIB, GIB)
    for rid in ('old-a', 'old-b', 'old-c'):
        assert budget.reserve(rid, e, 225*GIB, 240*GIB)
        budget.mark_decoding(rid)
    assert budget.reserved_bytes() == 5*GIB
    incoming = RequestMemoryEnvelope(324000, GIB, 5*GIB, GIB, 6*GIB)
    assert 225*GIB + 3*(GIB+2*GIB) + (6*GIB+2*GIB) > 238*GIB
    assert budget.reserve('head', incoming, 225*GIB, 240*GIB)
    assert budget.last_decision['predicted_peak_bytes'] == 236*GIB
    snap = budget.snapshot(225*GIB, 240*GIB)
    assert snap['reserved_bytes'] == sum(r['reserved_bytes'] for r in snap['reservations']) + snap['shared_transient_bytes']
    assert budget.decode_additional_bytes(['old-a', 'old-b', 'old-c']) == 5*GIB
    assert not budget.reserve('too-large', incoming, 225*GIB, 240*GIB)


def test_first_request_pays_shared_workspace_and_blocker_reports_marginal_cost():
    budget = RequestMemoryBudget()
    budget.shared_reclaim_bytes = 2*GIB
    e = RequestMemoryEnvelope(500000, GIB, 4*GIB, GIB, 5*GIB)
    assert not budget.reserve('head', e, 232*GIB, 240*GIB)
    snap = budget.snapshot(232*GIB, 240*GIB, 'head')
    assert snap['pending']['required_bytes'] == 7*GIB
    assert snap['pending']['shortfall_bytes'] == GIB
    budget.shared_reclaim_bytes = 0  # actual allocator tracker confirmed reallocation
    assert budget.snapshot(232*GIB, 240*GIB, 'head')['pending']['required_bytes'] == 5*GIB
    assert budget.reserve('head', e, 232*GIB, 240*GIB)


def test_materialized_credit_keeps_full_future_growth_and_two_copy_allowances():
    budget = RequestMemoryBudget()
    e = RequestMemoryEnvelope(500000, 3*GIB, 2*GIB, 4*GIB, 5*GIB)
    assert budget.reserve('one', e, 225*GIB, 240*GIB)
    budget.note_materialized('one', {10:GIB//4})
    snap = budget.snapshot(225*GIB, 240*GIB)
    row = snap['reservations'][0]
    assert snap['current_bytes'] == 225*GIB  # raw footprint is never reduced
    assert row['remaining_kv_bytes'] == 3*GIB//4
    assert row['copy_reserve_bytes'] == 2*GIB
    assert row['reserved_bytes'] == 5*GIB-GIB//4
    budget.note_materialized('one', {10:4*GIB})
    assert budget.reserved_bytes() == 4*GIB  # capped at one full KV copy
    budget.mark_decoding('one')
    assert budget.reserved_bytes() == 3*GIB
    budget.release('one')
    assert budget.reserve('one', e, 225*GIB, 240*GIB)
    assert budget.reserved_bytes() == 5*GIB  # retries cannot reuse old credit


def test_aliased_registered_buffers_are_never_credited_twice():
    budget = RequestMemoryBudget()
    e = RequestMemoryEnvelope(500000, 3*GIB, 0, 3*GIB, 3*GIB)
    for rid in ('a','b'):
        assert budget.reserve(rid,e,220*GIB,240*GIB)
        budget.note_materialized(rid,{123:GIB//2})
    assert budget.reserved_bytes() == 6*GIB-GIB//2
    budget.release('a')
    assert budget.reserved_bytes() == 3*GIB-GIB//2


def test_cache_credit_uses_registered_backing_without_materializing_state():
    from omlx.request_memory_budget import materialized_kv_buffers
    class DeepseekV41Cache:
        @property
        def cache(self): raise AssertionError('materialized row cache')
        @property
        def state(self): raise AssertionError('read lazy state')
    row = DeepseekV41Cache()
    view, buffer = object(), SimpleNamespace(nbytes=1024)
    row._dense = [None, None, view, None, None, None, None]
    row._ds41_buffers = {2:{'view':view,'buffer':buffer,'idle':SimpleNamespace(nbytes=4096)}}
    batch = DeepseekV41Cache();batch._rows=[row,row]
    assert materialized_kv_buffers([batch]) == {id(buffer):1024}
    row._dense[2] = object()  # stale growth registry after cache replacement
    assert materialized_kv_buffers([batch]) == {}
    assert materialized_kv_buffers([SimpleNamespace(state='unknown')]) == {}


def test_shared_split_preserves_existing_chunk_guard_prediction(scheduler, monkeypatch):
    monkeypatch.setattr(Scheduler,'_prefill_flat_overhead_enabled',lambda self:False)
    scheduler.memory_monitor.estimate_chunk_transient_bytes = lambda *a, **kw:100
    scheduler.memory_monitor.estimate_prompt_kv_bytes = lambda *a, **kw:10
    tracker = scheduler._prefill_transient_tracker
    tracker.update(5,500)
    tracker.record_reclaim(1000)
    old_guard = Scheduler._predicted_chunk_transient(scheduler,5,1000)
    local = Scheduler._predicted_chunk_transient(scheduler,5,1000,include_shared_reclaim=False)
    scheduler._refresh_pipeline_shared_budget()
    assert old_guard == 1143
    assert local == 650
    assert scheduler._request_memory_budget.shared_reclaim_bytes == 1000
    assert local + scheduler._request_memory_budget.shared_reclaim_bytes >= old_guard
    tracker.clear_reclaim()
    scheduler._refresh_pipeline_shared_budget()
    assert scheduler._request_memory_budget.shared_reclaim_bytes == 0


def test_new_shared_reclaim_is_checked_before_decoding(scheduler):
    scheduler._current_usage_bytes = lambda **kw:225*GIB
    request = Request(request_id='running',prompt=[1],sampling_params=SamplingParams(max_tokens=1))
    scheduler.add_request(request)
    assert scheduler._reserve_pipeline_memory(request)
    scheduler.waiting.clear();scheduler.running['running']=request
    assert scheduler._pipeline_decode_fits()
    scheduler._prefill_transient_tracker.record_reclaim(14*GIB)
    assert not scheduler._pipeline_decode_fits()


def test_decode_does_not_recharge_prefill_workspace_but_next_prefill_does():
    b = RequestMemoryBudget()
    b.separate_decode_workspace = True
    b.shared_reclaim_bytes = 2*GIB
    e = RequestMemoryEnvelope(100, 3*GIB, GIB, 4*GIB, 4*GIB)
    assert b.reserve('old', e, 230*GIB, 240*GIB)
    b.mark_decoding('old')
    # Restored memory is already in current usage; prefill debt stays intact
    # for a future prefill but is not copied into the independently bounded decode.
    assert 233*GIB+b.decode_additional_bytes(['old']) == 237*GIB
    assert b.shared_reclaim_bytes == 2*GIB
    assert not b.reserve('new', e, 233*GIB, 240*GIB)
    assert b.last_decision['required_bytes'] == 6*GIB
    assert b.snapshot(233*GIB,240*GIB)['shared_transient_bytes'] == 0


def test_native_decode_bound_covers_async_backbone_and_draft(scheduler, monkeypatch):
    monkeypatch.setenv('DS41_DECODE_ASYNC','4')
    monkeypatch.setenv('DS41_DRAFT_ASYNC','1')
    scheduler.memory_monitor.prefill_prediction_components=MagicMock(return_value={'profile':'deepseek_v41'})
    scheduler.memory_monitor.estimate_chunk_transient_bytes=MagicMock(return_value=100)
    scheduler.memory_monitor.estimate_prompt_kv_bytes=MagicMock(return_value=10)
    scheduler._current_usage_bytes=lambda **kw:220*GIB
    r=Request(request_id='bound',prompt=[1],sampling_params=SamplingParams(max_tokens=100))
    assert scheduler._reserve_pipeline_memory(r)
    b=scheduler._request_memory_budget
    assert b.separate_decode_workspace
    e=b.reservations['bound']
    assert e.decode_bytes-e.resident_bytes == 715  # 110 * 1.3 * (4 + 1)


def test_growth_credit_refresh_uses_uid_rows_without_materializing(scheduler):
    def forbidden(*args):raise AssertionError('Lazy cache materialized')
    cache_type=type('DeepseekV41Cache',(),{'cache':property(forbidden),'state':property(forbidden)})
    def row(size):
        c=cache_type();v=object();buffer=SimpleNamespace(nbytes=size)
        c._dense=[None,None,v,None,None,None,None]
        c._ds41_buffers={2:{'view':v,'buffer':buffer}}
        return c
    a,b=row(10),row(20)
    batchcache=cache_type();batchcache._rows=[b,a]
    scheduler.batch_generator=SimpleNamespace(_generation_batch=SimpleNamespace(uids=[22,11],prompt_cache=[batchcache]))
    scheduler.uid_to_request_id={11:'old',22:'new'}
    scheduler.running={'old':object(),'new':object()}
    budget=scheduler._request_memory_budget
    budget.reservations={rid:RequestMemoryEnvelope(100,300,0,400,400) for rid in scheduler.running}
    scheduler._record_completed_decode_kv()
    assert budget._credits()=={'old':10,'new':20}
    a._ds41_buffers[2]['buffer']=SimpleNamespace(nbytes=40)
    scheduler._record_completed_decode_kv()
    assert budget._credits()=={'old':40,'new':20}
    budget.release('old')
    assert budget._credits()=={'new':20}


def test_recovery_waits_then_preserves_old_decoders_and_emits_one_error(scheduler, monkeypatch):
    from omlx.scheduler import SchedulerOutput
    from omlx.request import RequestStatus
    now=SimpleNamespace(value=100.0)
    monkeypatch.setattr(scheduler_module.time,'monotonic',lambda:now.value)
    clear=MagicMock();monkeypatch.setattr(Scheduler,'_clear_cache',clear)
    for i in range(4):
        r=Request(request_id=str(i),prompt=[1],sampling_params=SamplingParams(max_tokens=100))
        r.arrival_time=float(i);r.status=RequestStatus.RUNNING
        scheduler.requests[r.request_id]=r;scheduler.running[r.request_id]=r
    current=SimpleNamespace(value=239*GIB)
    scheduler._current_usage_bytes=lambda **kw:current.value
    budget=scheduler._request_memory_budget
    budget.reservations={rid:RequestMemoryEnvelope(100,0,0,GIB,GIB) for rid in scheduler.running}
    out=SchedulerOutput()
    assert not scheduler._decode_budget_gate(out)
    assert len(scheduler.running)==4 and not out.outputs
    assert clear.call_count==1
    now.value=104.99
    assert not scheduler._decode_budget_gate(out)
    assert not out.outputs
    now.value=105.25
    assert not scheduler._decode_budget_gate(out)
    assert list(scheduler.running)==['0','1','2']
    assert [r.request_id for r in out.outputs]==['3']
    assert out.outputs[0].error_code=='decode_memory_budget_exceeded'
    assert len(scheduler._decode_budget_recovery.history)>=4
    assert scheduler._decode_budget_recovery.last_failure['victim_request_id']=='3'
    # The next call cannot kill another request using the same stale sample.
    now.value=105.30
    assert not scheduler._decode_budget_gate(out)
    assert len(out.outputs)==1
    current.value=230*GIB;now.value=105.60
    assert scheduler._decode_budget_gate(out)
    assert scheduler._decode_budget_recovery.state=='running'
    assert list(scheduler.running)==['0','1','2']


def test_recovery_resumes_after_delayed_physical_release_without_victim(scheduler, monkeypatch):
    from omlx.scheduler import SchedulerOutput
    now=SimpleNamespace(value=0.0)
    monkeypatch.setattr(scheduler_module.time,'monotonic',lambda:now.value)
    monkeypatch.setattr(Scheduler,'_clear_cache',MagicMock())
    r=Request(request_id='old',prompt=[1],sampling_params=SamplingParams(max_tokens=100))
    scheduler.running={'old':r};scheduler.requests={'old':r}
    scheduler._request_memory_budget.reservations={'old':RequestMemoryEnvelope(100,0,0,GIB,GIB)}
    current=SimpleNamespace(value=238*GIB);scheduler._current_usage_bytes=lambda **kw:current.value
    out=SchedulerOutput();assert not scheduler._decode_budget_gate(out)
    current.value=236*GIB;now.value=.3
    assert scheduler._decode_budget_gate(out)
    assert not out.outputs and 'old' in scheduler.running
    assert scheduler._decode_budget_recovery.history[-1]['stage']=='recovered'


def test_recovery_diagnostics_are_immutable_and_bounded():
    from omlx.request_memory_budget import DecodeBudgetRecovery
    g=DecodeBudgetRecovery();data={'requests':[{'request_id':'x'}],'current_bytes':123}
    for i in range(100):g.record('wait',data,float(i))
    data['requests'][0]['request_id']='changed'
    assert len(g.history)==64 and g.last_failure['requests'][0]['request_id']=='x'
    assert g.last_failure['event_id']==100


def test_decode_credit_runs_after_existing_sync_fence(scheduler, monkeypatch):
    calls=[]
    monkeypatch.setattr(scheduler_module,'_sync_and_clear_cache',lambda stream:calls.append('sync'))
    monkeypatch.setattr(Scheduler,'_record_completed_decode_kv',lambda self:calls.append('credit'))
    Scheduler._clear_cache(scheduler)
    assert calls==['sync','credit']


def test_row_removal_preserves_survivor_backings_without_dense_materialization(monkeypatch):
    from omlx.patches.deepseek_v41.cache import DeepseekV41Cache
    monkeypatch.setattr(DeepseekV41Cache,'_materialize',lambda self:pytest.fail('Materialized full KV during removal'))
    c=DeepseekV41Cache.__new__(DeepseekV41Cache)
    rows=[object(),object(),object(),object()]
    c._rows=rows;c._dense=None;c._lp=None;c._len=None
    c.filter([0,1,2])
    assert len(c._rows)==3 and all(a is b for a,b in zip(c._rows,rows))
    assert c._dense is None


def test_cancel_all_during_recovery_clears_pause(scheduler, monkeypatch):
    from omlx.scheduler import SchedulerOutput
    from omlx.request_memory_budget import DecodeBudgetRecovery
    admission=MagicMock()
    monkeypatch.setattr(scheduler_module,'get_request_admission',lambda:admission)
    scheduler._decode_budget_paused_models={'model'}
    scheduler._decode_budget_recovery.decide(False,1)
    scheduler.running={}
    assert scheduler._decode_budget_gate(SchedulerOutput())
    admission.pause.assert_called_with('model',False)
    assert scheduler._decode_budget_recovery.state=='running'
    assert not scheduler._decode_budget_paused_models


def allocation_row(end=100, capacity=256, idle_capacity=256, ratio=2):
    cls = type("DeepseekV41Cache", (), {})
    cache = cls()
    cache.compress_ratio = ratio
    view = object()
    cache._rows = None
    cache._dense = [None, None, view, None, None, None, None]
    def bank(n):
        return SimpleNamespace(shape=(1, n, 8), nbytes=n*8)
    cache._ds41_buffers = {2: {"view": view, "buffer": bank(capacity),
                              "idle": bank(idle_capacity) if idle_capacity else None,
                              "end": end}}
    return cache


@pytest.mark.parametrize("span", [1, 5, 17, 129])
def test_allocation_boundary_and_speculative_span(span):
    from omlx.request_memory_budget import allocation_layout
    from omlx.kv_capacity import growth_capacity
    cache = allocation_row(end=255)
    current, target, extra = allocation_layout([cache], 510, span)
    rows = (span+1)//2
    expected = 256 if rows == 1 else growth_capacity(255+rows, rows)
    assert current == 2*256*8
    assert target == expected*8
    assert extra == target + 2*rows*8


def test_within_capacity_only_reserves_current_copy_not_maximum_context():
    from omlx.request_memory_budget import allocation_layout
    cache = allocation_row()
    assert allocation_layout([cache], 200, 5) == (4096, 2048, 2096)
    # Rollback retains allocated banks; shortening logical length cannot free them.
    cache._ds41_buffers[2]['end'] = 50
    assert allocation_layout([cache], 100, 5) == (4096, 2048, 2096)


def test_credit_is_only_for_completed_live_banks_and_deduplicates_shared():
    from omlx.request_memory_budget import materialized_kv_buffers
    cache = allocation_row()
    monitor = SimpleNamespace(estimate_resident_kv_bytes=lambda n, **kw:n*8)
    e = estimate_envelope(monitor, 200, 999999, 8192, lambda q,k:100,
                          caches=[cache], decoding=True)
    b = RequestMemoryBudget()
    for rid in ('a','b'):
        b.reservations[rid] = e
        b.note_materialized(rid, materialized_kv_buffers([cache], include_idle=True))
    assert sum(b._credits().values()) == 4096
    cache._ds41_buffers[2]['idle'] = SimpleNamespace(shape=(1,256,8), nbytes=2048)
    b.refresh('a', e, materialized_kv_buffers([cache], include_idle=True))
    assert b._credits()['a'] == 2048  # replacement is not credited before a fence


def test_scheduler_refreshes_actual_progress_without_output_limit(scheduler):
    request = Request(request_id='progress', prompt=[1], sampling_params=SamplingParams(max_tokens=512000))
    request.num_prompt_tokens = 200
    scheduler._current_usage_bytes = lambda **kw: 220*GIB
    assert scheduler._reserve_pipeline_memory(request)
    scheduler.running[request.request_id] = request
    scheduler.request_id_to_uid[request.request_id] = 11
    cache = allocation_row()
    scheduler.batch_generator = SimpleNamespace(_generation_batch=SimpleNamespace(uids=[11],prompt_cache=[cache]))
    request.output_token_ids.extend([1]*20)
    scheduler._pipeline_decode_fits()
    e = scheduler._request_memory_budget.reservations['progress']
    assert e.context_tokens == 220
    assert e.current_capacity_bytes == 4096
    request.sampling_params.max_tokens = 256000
    scheduler._pipeline_decode_fits()
    assert scheduler._request_memory_budget.reservations['progress'] == e


def test_unknown_cache_uses_current_geometry_without_output_maximum():
    monitor=SimpleNamespace(estimate_resident_kv_bytes=lambda n, **kw:n*8)
    a=estimate_envelope(monitor, 200, 1, 8192, lambda q,k:100, caches=[object()],decoding=True)
    b=estimate_envelope(monitor, 200, 512000, 8192, lambda q,k:100, caches=[object()],decoding=True)
    assert a == b and a.context_tokens == 200


def test_mtp_unaccepted_drafts_and_queue_are_budgeted(scheduler):
    from collections import deque
    r=Request(request_id='mtp',prompt=[1],sampling_params=SamplingParams(max_tokens=4))
    r.num_prompt_tokens=200
    scheduler.request_id_to_uid['mtp']=1
    state=SimpleNamespace(depth=4,controller=SimpleNamespace(max_depth=8),
                          drafts=SimpleNamespace(shape=(16,)),queue=deque([1,2,3]))
    scheduler.batch_generator=SimpleNamespace(_generation_batch=SimpleNamespace(
        uids=[1],_omlx_mtp_batch_state=SimpleNamespace(states={1:state})))
    e=scheduler._pipeline_progress_envelope(r,decoding=True)
    assert e.step_tokens == 17
    assert e.context_tokens == 203

@pytest.mark.parametrize('count,append', [(250,1),(255,5),(256,17),(4095,5)])
def test_planned_growth_bounds_actual_cpu_allocator(count, append):
    import mlx.core as mx
    from omlx.patches.deepseek_v41 import growth
    from omlx.patches.deepseek_v41.cache import DeepseekV41Cache
    from omlx.request_memory_budget import allocation_layout
    with mx.stream(mx.cpu):
        cache=DeepseekV41Cache(compress_ratio=1)
        values=mx.zeros((1,count,8),mx.uint8)
        cache[2]=growth.append(cache,2,mx.zeros((1,0,8),mx.uint8),values,0)
        mx.eval(cache[2])
        old=cache._ds41_buffers[2]['buffer']
        plan=allocation_layout([cache],count,append)
        assert plan is not None
        cache[2]=growth.append(cache,2,cache[2],mx.ones((1,append,8),mx.uint8),count)
        mx.eval(cache[2])
        new=cache._ds41_buffers[2]['buffer']
        assert new.nbytes <= plan[2]
        assert old.nbytes + new.nbytes <= plan[0] + plan[2]


def test_non_source_empty_placeholders_do_not_disable_live_geometry():
    from omlx.request_memory_budget import allocation_layout
    source=allocation_row()
    empty=allocation_row(ratio=0)
    empty._ds41_buffers={}
    empty._dense[2]=SimpleNamespace(shape=(1,0,288),nbytes=0)
    empty._dense[3]=SimpleNamespace(shape=(1,0,68),nbytes=0)
    assert allocation_layout([empty,source,empty],200,5)==allocation_layout([source],200,5)


def test_rollback_view_keeps_old_banks_charged_and_requires_new_allocation():
    from omlx.request_memory_budget import allocation_layout, materialized_kv_buffers
    from omlx.kv_capacity import growth_capacity
    cache=allocation_row()
    cache._dense[2]=SimpleNamespace(shape=(1,90,8),nbytes=720)
    current,target,extra=allocation_layout([cache],180,5)
    assert current==4096
    assert target==growth_capacity(103,3)*8
    assert extra>=target
    assert sum(materialized_kv_buffers([cache],include_idle=True).values())==current


def test_idle_repair_lag_is_counted_even_after_depth_shrinks():
    from omlx.request_memory_budget import allocation_layout
    cache=allocation_row(end=200)
    cache._ds41_buffers[2]['valid']=150
    assert allocation_layout([cache],400,1)[2]==256*8+2*51*8


@pytest.fixture
def adaptive_scheduler(scheduler, monkeypatch):
    """Replay three GiB of spendable headroom without allocating a model."""
    current = int(221.91 * GIB)
    monkeypatch.setattr(scheduler_module, 'get_phys_footprint', lambda: current)
    monkeypatch.setattr(scheduler_module, 'get_macos_vm_stats', lambda: {'free': 3*GIB})
    scheduler._current_usage_bytes = lambda **kw: current
    scheduler._prefill_min_chunk_tokens = 256
    scheduler.memory_monitor.prefill_prediction_components = lambda *a: {'profile': 'deepseek_v41'}
    scheduler.memory_monitor.estimate_chunk_transient_bytes = lambda q,k: q * k * 20
    scheduler.memory_monitor.estimate_prompt_kv_bytes = lambda n: n * 890
    scheduler._predicted_chunk_transient = lambda q,k,**kw: int((.5 + q/8192*5) * GIB)
    return scheduler


@pytest.mark.parametrize('tokens', [198241, 371057])
def test_idle_long_context_admission_shrinks_before_queue_can_stall(adaptive_scheduler, tokens):
    s = adaptive_scheduler
    req = Request(request_id='head', prompt=[1], sampling_params=SamplingParams(max_tokens=262144))
    req.num_prompt_tokens = tokens
    full = s._pipeline_progress_envelope(req)
    assert not s._request_memory_budget.reserve(req.request_id, full,
                                               s._current_usage_bytes(), s._pipeline_system_limit())
    assert s._reserve_pipeline_memory(req)
    budget = s._request_memory_budget
    selected = budget.reservations['head']
    assert 256 <= selected.prefill_chunk_tokens < 8192
    assert selected.resident_bytes == full.resident_bytes
    assert selected.decode_bytes == full.decode_bytes
    assert budget.last_decision['predicted_peak_bytes'] <= int(224.91*GIB)
    assert s.config.prefill_step_size == 8192
    # Even if the physical guard could spend more, execution obeys this
    # request's reserved workspace. The short final slice is never enlarged.
    s._admission_transient_bound = lambda *a,**kw: 0
    for requested in (8192, 4096, 11):
        actual = s._guard_prefill_chunk(requested, kv_len=tokens, progress=0,
                                        loop_label='test', request_id='head')
        assert actual == min(requested, selected.prefill_chunk_tokens)
    assert budget.snapshot(s._current_usage_bytes(), 240*GIB)['reservations'][0]['prefill_chunk_tokens'] == selected.prefill_chunk_tokens


def test_adaptive_admission_retains_old_decode_and_shared_reserves(adaptive_scheduler):
    s = adaptive_scheduler
    b = s._request_memory_budget
    old = Request(request_id='old', prompt=[1], sampling_params=SamplingParams(max_tokens=262144))
    s.running['old'] = old
    b.reservations['old'] = RequestMemoryEnvelope(200000, 0, 0, 2*GIB, 2*GIB)
    b.mark_decoding('old')
    s._refresh_pipeline_shared_budget = lambda: None
    b.shared_reclaim_bytes = GIB
    req = Request(request_id='new', prompt=[1], sampling_params=SamplingParams(max_tokens=262144))
    req.num_prompt_tokens = 371057
    assert not s._reserve_pipeline_memory(req)
    assert b.last_decision['reserved_bytes'] == 2*GIB
    assert b.last_decision['envelope']['prefill_chunk_tokens'] == 256
    assert b.last_decision['required_bytes'] >= GIB + b.last_decision['envelope']['total_bytes']
    assert 'new' not in b.reservations
    assert s._pipeline_decode_fits()
    assert list(s.running) == ['old']


def test_full_prefill_width_preserved_when_it_fits(adaptive_scheduler, monkeypatch):
    s = adaptive_scheduler
    monkeypatch.setattr(scheduler_module, 'get_macos_vm_stats', lambda: {'free': 24*GIB})
    req = Request(request_id='wide', prompt=[1], sampling_params=SamplingParams(max_tokens=262144))
    req.num_prompt_tokens = 371057
    assert s._reserve_pipeline_memory(req)
    assert s._request_memory_budget.reservations['wide'].prefill_chunk_tokens == 8192


def test_released_reservation_does_not_leave_a_stale_prefill_width(adaptive_scheduler):
    s = adaptive_scheduler
    req = Request(request_id='reuse', prompt=[1], sampling_params=SamplingParams(max_tokens=16))
    req.num_prompt_tokens = 371057
    assert s._reserve_pipeline_memory(req)
    s._request_memory_budget.release('reuse')
    s._admission_transient_bound = lambda *a, **kw: 0
    assert s._guard_prefill_chunk(8192, kv_len=0, progress=0, loop_label='test', request_id='reuse') == 8192


def test_incompatible_execution_floor_cannot_exceed_reserved_workspace(adaptive_scheduler):
    from omlx.exceptions import PrefillMemoryExceededError
    s = adaptive_scheduler
    req = Request(request_id='floor', prompt=[1], sampling_params=SamplingParams(max_tokens=16))
    req.num_prompt_tokens = 371057
    assert s._reserve_pipeline_memory(req)
    with pytest.raises(PrefillMemoryExceededError, match='Rate limit'):
        s._guard_prefill_chunk(8192, kv_len=0, progress=0, loop_label='test',
                              request_id='floor', minimum_tokens=8192)


@pytest.mark.parametrize('queued', [0, 1, 3, 4, 8])
def test_mtp_verified_queue_advances_context_not_verify_workspace(scheduler, queued):
    from collections import deque
    s = scheduler
    r = Request(request_id='buffered', prompt=[1], sampling_params=SamplingParams(max_tokens=262144))
    r.num_prompt_tokens = 221647
    r.output_token_ids.extend([1]*2011)
    s.request_id_to_uid[r.request_id] = 1
    state = SimpleNamespace(depth=4, controller=SimpleNamespace(max_depth=4),
                            drafts=SimpleNamespace(shape=(4,)), queue=deque([1]*queued))
    s.batch_generator = SimpleNamespace(_generation_batch=SimpleNamespace(uids=[1], _omlx_mtp_state=state))
    calls = []
    s.memory_monitor.prefill_prediction_components = lambda *a: {'profile': 'deepseek_v41'}
    s.memory_monitor.estimate_prompt_kv_bytes = lambda n: n*890
    s.memory_monitor.estimate_chunk_transient_bytes = lambda q,k: calls.append((q,k)) or (GIB if q>8 else 16*1024**2)
    e = s._pipeline_progress_envelope(r, decoding=True)
    assert e.step_tokens == 5
    assert e.context_tokens == 221647 + 2011 + queued
    assert calls[-1][0] == 5
    assert e.decode_bytes - e.resident_bytes < GIB
    assert e.prefill_chunk_tokens == 0


def test_prefill_guard_reprices_system_limit_after_measured_reclaim(adaptive_scheduler, monkeypatch):
    s=adaptive_scheduler
    req=Request(request_id='released',prompt=[1],sampling_params=SamplingParams(max_tokens=16))
    req.num_prompt_tokens=198241
    assert s._reserve_pipeline_memory(req)
    s.requests[req.request_id]=req
    physical={'current':int(223.2*GIB),'free':int(.5*GIB)}
    monkeypatch.setattr(scheduler_module,'get_phys_footprint',lambda:physical['current'])
    monkeypatch.setattr(scheduler_module,'get_macos_vm_stats',lambda:{'free':physical['free']})
    s._current_usage_bytes=lambda **kw:physical['current']
    s._admission_transient_bound=lambda *a,**kw:int(.75*GIB)
    s._predicted_chunk_transient=lambda *a,**kw:int(.75*GIB)
    def reclaim():
        physical.update(current=int(222.8*GIB),free=3*GIB)
        return physical['current']
    s._reclaim_prefill_headroom=reclaim
    assert s._guard_prefill_chunk(2048,kv_len=62432,progress=62432,
                                 loop_label='chunked_step',request_id=req.request_id)>0
    assert s._last_pipeline_system_budget['system_free_bytes']==3*GIB
    assert getattr(req,'_prefill_budget_wait_started',None) is None


def test_prefill_wait_preserves_cache_and_yields_executor_without_admitting_new_work(adaptive_scheduler,monkeypatch):
    from collections import deque
    s=adaptive_scheduler
    req=Request(request_id='waiting-prefill',prompt=[1],sampling_params=SamplingParams(max_tokens=16))
    req.num_prompt_tokens=198241
    assert s._reserve_pipeline_memory(req)
    s.requests[req.request_id]=req
    s.prefilling=deque([req]);s._decode_fairness=False
    state=SimpleNamespace(request=req,cache=[object()],tokens_processed=62432)
    s._prefill_states[req.request_id]=state
    now={'value':100.0};monkeypatch.setattr(scheduler_module.time,'monotonic',lambda:now['value'])
    calls=[]
    def step(st):
        calls.append(st)
        s._wait_for_prefill_budget(req.request_id,current=int(222.8*GIB),cap=int(223.49*GIB),
                                  transient=int(.77*GIB),kv_len=62432,chunk_tokens=32,resumable=True)
    s._step_prefill_chunk=step
    scheduled=[];rejected=[]
    assert not s._advance_chunked_prefills(scheduled,rejected)
    assert s._prefill_states[req.request_id] is state and state.tokens_processed==62432
    assert list(s.prefilling)==[req] and not rejected
    later=Request(request_id='later',prompt=[1],sampling_params=SamplingParams(max_tokens=1))
    assert not s._reserve_pipeline_memory(later)
    assert not s._advance_chunked_prefills(scheduled,rejected) and len(calls)==1
    now['value']=100.3
    assert not s._advance_chunked_prefills(scheduled,rejected) and len(calls)==2
    # Once physical room returns, reuse the exact state and clear the wait.
    s._admission_transient_bound=lambda *a,**kw:0
    s._step_prefill_chunk=lambda st: bool(s._guard_prefill_chunk(32,kv_len=62432,progress=62432,
                                      loop_label='chunked_step',request_id=req.request_id)) and False
    now['value']=100.6
    assert s._advance_chunked_prefills(scheduled,rejected)
    assert state.tokens_processed==62432 and s._prefill_states[req.request_id] is state
    assert req._prefill_budget_wait_started is None
    assert s._memory_budget_snapshot()['prefill_guard']['waiting_request_ids']==[]


def test_prefill_wait_timeout_only_terminates_affected_request_as_rate_limit(adaptive_scheduler,monkeypatch):
    from omlx.scheduler import _PrefillBudgetWait,_PrefillBudgetExceeded,_prefill_memory_exception_output
    from omlx.engine_core import _raise_request_output_error
    from omlx.request_memory_budget import TransientMemoryPressureError
    s=adaptive_scheduler
    req=Request(request_id='starved',prompt=[1],sampling_params=SamplingParams(max_tokens=16))
    s.requests[req.request_id]=req
    now={'value':100.0};monkeypatch.setattr(scheduler_module.time,'monotonic',lambda:now['value'])
    def check():s._wait_for_prefill_budget(req.request_id,current=223*GIB,cap=223*GIB,
                                         transient=GIB,kv_len=62432,chunk_tokens=32,resumable=True)
    with pytest.raises(_PrefillBudgetWait):check()
    now['value']=159.9
    with pytest.raises(_PrefillBudgetWait):check()
    # Old decode gets the opportunity to finish; no deadline kills its peer.
    s.running['old']=SimpleNamespace(request_id='old')
    now['value']=200.0
    with pytest.raises(_PrefillBudgetWait):check()
    s.running.clear();now['value']=260.1
    with pytest.raises(_PrefillBudgetExceeded)as info:check()
    output=_prefill_memory_exception_output(req.request_id,info.value)
    assert output.error_code=='prefill_memory_budget_exceeded'
    with pytest.raises(TransientMemoryPressureError,match='Rate limit:'):
        _raise_request_output_error(output)
    failure=s._last_prefill_budget_failure
    assert failure['stage']=='failed' and failure['shortfall_bytes']==GIB
    assert failure['kv_tokens']==62432 and failure['current_bytes']==223*GIB


@pytest.mark.parametrize('vision',[False,True])
def test_external_prefill_memory_wait_keeps_token_embedding_and_cache_cursors(adaptive_scheduler,monkeypatch,vision):
    import mlx.core as mx
    from test_scheduler import TestVLMPositionStateClearing
    s=adaptive_scheduler;s.model=TestVLMPositionStateClearing()._make_vlm_model()
    s.config.prefill_step_size=8
    req=Request(request_id='external',prompt=list(range(18)),sampling_params=SamplingParams(max_tokens=16))
    req.prompt_token_ids=list(range(18));req.num_prompt_tokens=18;s.requests[req.request_id]=req
    s.request_id_to_uid[req.request_id]=7;s.uid_to_request_id[7]=req.request_id
    s._adaptive_chunk_size=lambda n,**kw:n
    s._check_post_prefill_memory=lambda **kw:None
    assert s._reserve_pipeline_memory(req)
    attempted=[];sleeps=[];cache=[]
    def guard(n,**kw):
        attempted.append(kw['progress'])
        if attempted==[0,8]:
            s._wait_for_prefill_budget(req.request_id,current=223*GIB,cap=223*GIB,
                                      transient=GIB,kv_len=8,chunk_tokens=8,resumable=True,yielding=False)
        return s._record_prefill_chunk_decision(n,n,kw['kv_len'],gathered_core=False,
                                               request_id=req.request_id,stage='pre_chunk_accepted')
    s._guard_prefill_chunk=guard
    monkeypatch.setattr(scheduler_module.time,'sleep',sleeps.append)
    embeds=mx.arange(18).reshape(1,18,1) if vision else None
    result,last=s._do_external_prefill(req,req.prompt_token_ids,cache,(embeds,{},0)if vision else None)
    assert attempted==[0,8,8,16] and sleeps==[.25]
    assert result is cache and last==[17]
    assert [call.args[0].tolist()[0]for call in s.model.call_args_list]==[list(range(8)),list(range(8,16)),[16]]
    if vision:
        assert [call.kwargs['inputs_embeds'].reshape(-1).tolist()for call in s.model.call_args_list]==[list(range(8)),list(range(8,16)),[16]]
    assert req._prefill_budget_wait_started is None
    assert s._admin_snapshot['memory_budget']['prefill_guard']['waiting_request_ids']==[]


@pytest.mark.parametrize('vision',[False,True])
def test_external_prefill_cancel_during_memory_wait_does_not_run_another_chunk(adaptive_scheduler,monkeypatch,vision):
    import mlx.core as mx
    from test_scheduler import TestVLMPositionStateClearing
    from omlx.scheduler import _PrefillAbortedError
    s=adaptive_scheduler;s.model=TestVLMPositionStateClearing()._make_vlm_model()
    s.config.prefill_step_size=8
    req=Request(request_id='cancel-wait',prompt=list(range(18)),sampling_params=SamplingParams(max_tokens=16))
    req.prompt_token_ids=list(range(18));req.num_prompt_tokens=18;s.requests[req.request_id]=req
    s.request_id_to_uid[req.request_id]=7;s.uid_to_request_id[7]=req.request_id
    s._adaptive_chunk_size=lambda n,**kw:n
    s._check_post_prefill_memory=lambda **kw:None
    assert s._reserve_pipeline_memory(req)
    def guard(n,**kw):
        if kw['progress']==8:
            s.abort_request(req.request_id)
            s._wait_for_prefill_budget(req.request_id,current=223*GIB,cap=223*GIB,
                                      transient=GIB,kv_len=8,chunk_tokens=8,resumable=True,yielding=False)
        return n
    s._guard_prefill_chunk=guard
    monkeypatch.setattr(scheduler_module.time,'sleep',lambda _:pytest.fail('Cancellation should not sleep'))
    embeds=mx.arange(18).reshape(1,18,1)if vision else None
    with pytest.raises(_PrefillAbortedError)as error:
        s._do_external_prefill(req,req.prompt_token_ids,[],(embeds,{},0)if vision else None)
    assert error.value.processed_tokens==8 and s.model.call_count==1
    s._cleanup_prefill_abort_request(req,temp_uid=7)
    assert req.request_id not in s.requests and req.request_id not in s._request_memory_budget.reservations


def test_external_wait_cannot_indefinitely_block_an_existing_decoder(adaptive_scheduler,monkeypatch):
    from omlx.scheduler import _PrefillBudgetWait,_PrefillBudgetExceeded
    s=adaptive_scheduler;req=Request(request_id='external-deadline',prompt=[1],sampling_params=SamplingParams(max_tokens=16))
    s.requests[req.request_id]=req;s.running['old']=SimpleNamespace(request_id='old')
    now={'value':100.};monkeypatch.setattr(scheduler_module.time,'monotonic',lambda:now['value'])
    def check():s._wait_for_prefill_budget(req.request_id,current=223*GIB,cap=223*GIB,
                                         transient=GIB,kv_len=8,chunk_tokens=8,resumable=True,yielding=False)
    with pytest.raises(_PrefillBudgetWait):check()
    now['value']=105.1
    with pytest.raises(_PrefillBudgetExceeded,match='Rate limit:'):check()
    assert list(s.running)==['old']
