"""Mocked CPU scheduler replay of preferred/emergency admission and live upgrades."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from test_pipeline_memory_budget import scheduler, system_telemetry
from omlx.request import Request, SamplingParams
from omlx.request_memory_budget import GIB, RequestMemoryEnvelope
import omlx.scheduler as module

MIB = 1024**2


@pytest.fixture
def tiers(scheduler, monkeypatch):
    s = scheduler
    s._prefill_min_chunk_tokens = 256
    s._prefill_preferred_chunk_tokens = 512
    s._request_memory_budget.separate_decode_workspace = True
    s._current_usage_bytes = lambda **kw: 222 * GIB
    s._reclaim_prefill_headroom = lambda: 222 * GIB
    s._admission_limit_bytes = lambda: 238 * GIB
    s._test_limit = int(223.4 * GIB)
    s._prefill_peak_target = lambda: s._test_limit
    s._prefill_abort_description = lambda: (240 * GIB, s._test_limit, .995)
    s._refresh_pipeline_shared_budget = lambda: None
    s._refresh_decode_progress_budgets = lambda: None
    s._admission_transient_bound = lambda q, k, **kw: q * MIB
    s._predicted_chunk_transient = lambda q, k, **kw: q * MIB
    s._pipeline_progress_envelope = lambda request, prefill_step_size=None, **kw: envelope(
        min(prefill_step_size or 8192, request.num_prompt_tokens - 1))
    s._test_now = 100.0
    monkeypatch.setattr(module.time, 'monotonic', lambda: s._test_now)
    return s


def envelope(q):
    return RequestMemoryEnvelope(200000, GIB, q*MIB, GIB+10*MIB, GIB+q*MIB,
                                 credit_cap_bytes=GIB, prefill_chunk_tokens=q)


def request(s, rid='head', tokens=200000):
    r = Request(request_id=rid, prompt=[1], sampling_params=SamplingParams(max_tokens=262144))
    r.num_prompt_tokens = tokens
    s.requests[rid] = r
    return r


def guard(s, r, *, progress=256, requested=8192, path='chunked_step'):
    return s._guard_prefill_chunk(requested, kv_len=progress, progress=progress,
                                 loop_label=path, request_id=r.request_id)


def test_idle_head_uses_safe_emergency_floor_instead_of_idle_queue(tiers):
    r = request(tiers)
    assert tiers._reserve_pipeline_memory(r)
    assert tiers._request_memory_budget.reservations[r.request_id].prefill_chunk_tokens == 256
    assert tiers._request_memory_budget.last_decision['predicted_peak_bytes'] <= tiers._test_limit


@pytest.mark.parametrize('kind', ['decode', 'prefill', 'cleanup'])
def test_new_head_waits_for_releasing_work_instead_of_small_chunk(tiers, kind):
    r = request(tiers)
    old = request(tiers, 'old')
    if kind == 'decode': tiers.running['old'] = old
    elif kind == 'prefill': tiers.prefilling.append(old)
    else: tiers._pending_async_removes.append(SimpleNamespace(request_id='old'))
    assert not tiers._reserve_pipeline_memory(r)
    assert r.request_id not in tiers._request_memory_budget.reservations
    assert tiers._request_memory_budget.last_decision['envelope']['prefill_chunk_tokens'] == 512


def test_admission_prefers_largest_fitting_normal_tier(tiers):
    r = request(tiers)
    tiers._test_limit = 225 * GIB
    assert tiers._reserve_pipeline_memory(r)
    assert tiers._request_memory_budget.reservations[r.request_id].prefill_chunk_tokens == 2048


def test_admission_never_drops_below_emergency_to_force_a_fit(tiers):
    r = request(tiers)
    tiers._test_limit = int(223.1 * GIB)
    assert not tiers._reserve_pipeline_memory(r)
    assert not tiers._request_memory_budget.reservations
    assert tiers._request_memory_budget.last_decision['envelope']['prefill_chunk_tokens'] == 256


@pytest.mark.parametrize('path', ['external', 'chunked_step'])
def test_upgrade_only_after_progress_and_one_tier_per_cooldown(tiers, path):
    r = request(tiers)
    assert tiers._reserve_pipeline_memory(r)
    tiers._test_limit = 230 * GIB
    assert guard(tiers, r, progress=0, path=path) == 256
    assert guard(tiers, r, progress=256, path=path) == 512
    tiers._test_now += .5
    assert guard(tiers, r, progress=768, path=path) == 512
    tiers._test_now += .6
    assert guard(tiers, r, progress=1280, path=path) == 1024
    tiers._test_now += 2
    assert guard(tiers, r, progress=1280, path=path) == 1024
    assert guard(tiers, r, progress=2304, path=path) == 2048
    events = tiers._memory_budget_snapshot()['prefill_policy']['events']
    assert [e['selected_chunk_tokens'] for e in events] == [512,1024,2048]
    assert all(e['predicted_peak_bytes']+e['headroom_margin_bytes'] <= e['limit_bytes'] for e in events)


def test_resize_rejection_keeps_old_plan_and_does_not_start_error_timer(tiers):
    r = request(tiers)
    assert tiers._reserve_pipeline_memory(r)
    previous = tiers._request_memory_budget.reservations[r.request_id]
    tiers._test_limit = int(223.5 * GIB)  # 512 fits exactly, but no upgrade margin.
    assert guard(tiers, r) == 256
    assert tiers._request_memory_budget.reservations[r.request_id] is previous
    assert getattr(r, '_prefill_budget_wait_started', None) is None
    assert not tiers._request_memory_budget.last_resize_decision['allowed']


@pytest.mark.parametrize('width', [11, 128, 256, 511])
def test_short_tail_boundary_and_decode_fairness_cap_are_never_enlarged(tiers, width):
    r = request(tiers)
    tiers._request_memory_budget.reservations[r.request_id] = envelope(512)
    tiers._test_limit = 230 * GIB
    assert guard(tiers, r, requested=width) == width
    assert not tiers._prefill_resize_events


@pytest.mark.parametrize('path', ['external', 'chunked_step'])
def test_inflight_256_progress_does_not_wait_for_preferred_512(tiers, path):
    r = request(tiers)
    tiers._request_memory_budget.reservations[r.request_id] = envelope(512)
    tiers.running['old'] = request(tiers, 'old')
    tiers._test_limit = 222 * GIB + 300 * MIB
    tiers._wait_for_prefill_budget = MagicMock(side_effect=AssertionError('performance preference became memory failure'))
    assert guard(tiers, r, requested=512, path=path) == 256
    tiers._wait_for_prefill_budget.assert_not_called()


def test_readmission_resets_upgrade_coordinate(tiers):
    r = request(tiers)
    r._prefill_upgrade_progress = 200000
    r._prefill_upgrade_after = 10000
    assert tiers._reserve_pipeline_memory(r)
    tiers._test_limit = 230 * GIB
    assert guard(tiers, r, progress=256) == 512


def test_disabled_policy_keeps_existing_admission_and_no_resize(tiers):
    tiers._prefill_preferred_chunk_tokens = 0
    r = request(tiers)
    tiers.running['old'] = request(tiers, 'old')
    assert tiers._reserve_pipeline_memory(r)
    tiers._test_limit = 230 * GIB
    assert guard(tiers, r) == 256
    assert not tiers._prefill_resize_events
