"""Pool-first DS41 reclaim at the real scheduler boundaries, with fake usage."""

from collections import OrderedDict
from types import SimpleNamespace

import pytest

import omlx.scheduler as module
from omlx.scheduler import Scheduler
from test_scheduler_exact_prefix_cache import _prefill, _request, _scheduler


class FakeExact:
    def __init__(self, sizes=(80, 80, 80), protected=()):
        self.entries = OrderedDict(enumerate(sizes))
        self.bytes = sum(sizes)
        self.protected = set(protected)
        self.evictions = []
        self.requests = []

    def evict_bytes(self, required):
        self.requests.append(required)
        freed = 0
        for key in list(self.entries):
            if freed >= required:
                break
            if key in self.protected:
                continue
            size = self.entries.pop(key)
            freed += size
            self.bytes -= size
            self.evictions.append(key)
        return freed


def harness(monkeypatch, *, live=700, pool=200, soft=1000,
            admission=1100, abort=1200, sizes=(80, 80, 80), protected=()):
    scheduler = Scheduler.__new__(Scheduler)
    exact = FakeExact(sizes, protected)
    state = {"live": live, "pool": pool, "clears": 0, "measurements": []}
    scheduler.model = SimpleNamespace()
    scheduler._pending_pressure_clear = False
    scheduler._memory_limit_bytes = soft
    scheduler._admission_limit_bytes = lambda: admission
    scheduler._prefill_abort_cap = lambda: abort

    def usage():
        state["measurements"].append(state["pool"])
        return state["live"] + exact.bytes + state["pool"]

    def clear(self):
        state["pool"] = 0
        state["clears"] += 1

    scheduler._current_usage_bytes = usage
    monkeypatch.setattr(module, "_exact_prefix_manager", lambda model: exact)
    monkeypatch.setattr(Scheduler, "_clear_cache", clear)
    return scheduler, exact, state


def drain(scheduler):
    scheduler.request_pressure_reclaim()
    assert scheduler._consume_pressure_clear()
    Scheduler._clear_cache(scheduler)
    scheduler._reclaim_exact_prefix_after_pressure_clear()


def test_consuming_pressure_request_does_not_discard_state(monkeypatch):
    scheduler, exact, state = harness(monkeypatch)
    scheduler.request_pressure_reclaim()
    assert scheduler._consume_pressure_clear()
    assert not scheduler._consume_pressure_clear()
    assert exact.bytes == 240
    assert exact.requests == []
    assert state["clears"] == 0


def test_soft_pool_drain_retains_every_checkpoint(monkeypatch):
    scheduler, exact, state = harness(monkeypatch)
    assert scheduler._current_usage_bytes() == 1140
    drain(scheduler)
    assert exact.bytes == 240
    assert list(exact.entries) == [0, 1, 2]
    assert exact.requests == []
    assert state["clears"] == 1
    assert state["measurements"][-1] == 0
    assert scheduler._current_usage_bytes() == 940


@pytest.mark.parametrize("soft,admission,abort", [
    (1000, 1100, 1200), (1200, 1000, 1100), (1200, 1100, 1000),
])
def test_residual_deficit_evicts_minimum_lru_snapshots(
    monkeypatch, soft, admission, abort,
):
    scheduler, exact, state = harness(
        monkeypatch, live=850, soft=soft, admission=admission, abort=abort,
    )
    drain(scheduler)
    assert exact.requests == [90]
    assert exact.evictions == [0, 1]
    assert list(exact.entries) == [2]
    assert scheduler._current_usage_bytes() == 930
    assert state["clears"] == 2


def test_hard_pressure_reclaims_all_if_residual_requires_it(monkeypatch):
    scheduler, exact, state = harness(monkeypatch, live=1100)
    drain(scheduler)
    assert exact.requests == [340]
    assert exact.bytes == 0
    # A deficit with no idle snapshots remains observable to the enforcer.
    # The scheduler does not change limits or pretend recovery succeeded.
    assert scheduler._current_usage_bytes() == 1100
    assert scheduler._memory_limit_bytes == 1000
    assert state["clears"] == 2


def test_protected_restore_snapshot_survives_residual_reclaim(monkeypatch):
    scheduler, exact, _ = harness(monkeypatch, live=850, protected=(0,))
    drain(scheduler)
    assert exact.evictions == [1, 2]
    assert list(exact.entries) == [0]
    assert scheduler._current_usage_bytes() == 930


def test_unconfigured_limits_keep_conservative_reclaim(monkeypatch):
    scheduler, exact, state = harness(monkeypatch, soft=0, admission=0, abort=0)
    drain(scheduler)
    assert exact.requests == [240]
    assert exact.bytes == 0
    assert state["clears"] == 2


def test_empty_exact_cache_needs_no_additional_clear(monkeypatch):
    scheduler, exact, state = harness(monkeypatch, sizes=())
    drain(scheduler)
    assert exact.requests == []
    assert state["clears"] == 1


def test_real_external_prefill_boundary_retains_reusable_prefix(
    mock_tokenizer, monkeypatch,
):
    scheduler = _scheduler(mock_tokenizer, monkeypatch)
    cold = _request("cold")
    scheduler._prepare_prefix_cache_for_request(cold)
    _prefill(scheduler, cold, False)
    exact = scheduler.model._omlx_exact_prefix_cache
    before = set(exact.entries)
    assert len(before) == 2
    clears = []
    monkeypatch.setattr(Scheduler, "_clear_cache", lambda self: clears.append(True))
    scheduler.request_pressure_reclaim()
    other = _request("other")
    other.prompt_token_ids[10] += 100
    scheduler._prepare_prefix_cache_for_request(other)
    _prefill(scheduler, other, False)
    assert not scheduler._pending_pressure_clear
    assert before <= set(exact.entries)
    warm = _request("warm")
    scheduler._prepare_prefix_cache_for_request(warm)
    assert warm.cached_tokens == 16384
    assert clears


def test_real_idle_step_pressure_boundary_retains_reusable_prefix(
    mock_tokenizer, monkeypatch,
):
    scheduler = _scheduler(mock_tokenizer, monkeypatch)
    cold = _request("cold")
    scheduler._prepare_prefix_cache_for_request(cold)
    _prefill(scheduler, cold, False)
    exact = scheduler.model._omlx_exact_prefix_cache
    before = set(exact.entries)
    clears = []
    monkeypatch.setattr(Scheduler, "_clear_cache", lambda self: clears.append(True))
    scheduler.request_pressure_reclaim()
    scheduler.step()
    assert not scheduler._pending_pressure_clear
    assert set(exact.entries) == before
    assert len(clears) == 1
    warm = _request("warm")
    scheduler._prepare_prefix_cache_for_request(warm)
    assert warm.cached_tokens == 16384
