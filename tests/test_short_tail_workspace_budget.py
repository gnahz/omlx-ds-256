"""Cache-aware workspace admission, using byte counters and no model forward."""

from contextlib import nullcontext
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import omlx.scheduler as scheduler_module
from omlx.request_memory_budget import GIB, RequestMemoryEnvelope, estimate_envelope
from test_whole_prefill_reservation import request, whole_scheduler


CHUNK = 8192
PROMPT = 66000
HIT = 65536
TAIL = 463
UNCHANGED_FIELDS = (
    "context_tokens", "resident_bytes", "decode_bytes", "credit_cap_bytes",
    "current_capacity_bytes", "next_capacity_bytes", "allocation_growth_bytes",
    "step_tokens",
)


def _assert_context_and_decode_unchanged(before, after):
    for name in UNCHANGED_FIELDS:
        assert getattr(after, name) == getattr(before, name), name


def _exact_request(scheduler, *, count=PROMPT, cached=HIT):
    head = request(scheduler, tokens=count)
    head.prompt_token_ids = [1] * count
    exact = SimpleNamespace(
        max_bytes=2 * GIB,
        preview_cached_tokens=MagicMock(return_value=cached),
    )
    scheduler.model._omlx_exact_prefix_cache = exact
    return head, exact


def _prepared(scheduler, head, count, *, valid_cache=True):
    head.prompt_cache = [object()] if count and valid_cache else None
    head.cached_tokens = count
    head.cache_reused_tokens = count
    head.remaining_tokens = head.prompt_token_ids[count:]
    scheduler._prefix_cache_prepared.add(head.request_id)


@pytest.mark.parametrize("tail", [0, 1, 8, 9, 463, 8192, 8193])
def test_envelope_changes_only_workspace_width_and_peak(tail):
    monitor = SimpleNamespace(
        estimate_resident_kv_bytes=lambda count, **kw: count * 890,
    )
    prefill = MagicMock(side_effect=lambda q, context: q * 10000 + context)
    decode = MagicMock(side_effect=lambda q, context: q * 100 + context)
    cold = estimate_envelope(monitor, PROMPT, 256000, CHUNK, prefill,
                             decode_predict=decode)
    prefill.reset_mock()
    warm = estimate_envelope(monitor, PROMPT, 256000, CHUNK, prefill,
                             decode_predict=decode, remaining_prefill_tokens=tail)

    _assert_context_and_decode_unchanged(cold, warm)
    assert warm.context_tokens == PROMPT
    assert warm.resident_bytes == 3 * warm.next_capacity_bytes
    assert warm.prefill_chunk_tokens == min(CHUNK, tail)
    if tail:
        prefill.assert_called_once_with(min(CHUNK, tail), PROMPT - min(CHUNK, tail))
    else:
        prefill.assert_not_called()
        assert warm.prefill_bytes == 0
        assert warm.total_bytes == warm.decode_bytes
    assert warm.total_bytes == max(warm.resident_bytes + warm.prefill_bytes,
                                   warm.decode_bytes)


def test_decode_envelope_ignores_short_tail_hint():
    monitor = SimpleNamespace(estimate_resident_kv_bytes=lambda count, **kw: count * 890)
    predict = lambda q, context: q * 10000 + context
    cold = estimate_envelope(monitor, PROMPT, 1, CHUNK, predict, decoding=True)
    warm = estimate_envelope(monitor, PROMPT, 256000, CHUNK, predict,
                             decoding=True, remaining_prefill_tokens=0)
    assert asdict(warm) == asdict(cold)


def test_scheduler_preview_prices_full_context_with_short_workspace(whole_scheduler):
    scheduler, _ = whole_scheduler
    head, exact = _exact_request(scheduler)
    exact.preview_cached_tokens.return_value = 0
    cold = scheduler._pipeline_progress_envelope(head)
    exact.preview_cached_tokens.return_value = HIT
    warm = scheduler._pipeline_progress_envelope(head)

    _assert_context_and_decode_unchanged(cold, warm)
    assert cold.prefill_chunk_tokens == CHUNK
    assert warm.prefill_chunk_tokens == TAIL
    assert warm.prefill_bytes < cold.prefill_bytes
    assert scheduler.config.prefill_step_size == CHUNK
    assert scheduler._prefill_whole_reservation is True
    assert head.prompt_token_ids == [1] * PROMPT
    assert head.num_prompt_tokens == PROMPT


@pytest.mark.parametrize("kind", ["paged", "sparse", "requeue", "length_mismatch"])
def test_ineligible_requests_keep_the_complete_workspace(whole_scheduler, kind):
    scheduler, _ = whole_scheduler
    head, exact = _exact_request(scheduler)
    if kind == "paged":
        scheduler.block_aware_cache = object()
    elif kind == "sparse":
        head.specprefill_indices = [1]
    elif kind == "requeue":
        head.requeued_output_tokens = 1
        head.prompt_token_ids.append(2)
    else:
        head.num_prompt_tokens += 1

    assert scheduler._remaining_exact_prefill_tokens(head) is None
    envelope = scheduler._pipeline_progress_envelope(head)
    assert envelope.prefill_chunk_tokens == CHUNK
    exact.preview_cached_tokens.assert_not_called()


@pytest.mark.parametrize("kind", ["unknown_profile", "adaptive_policy"])
def test_other_profiles_and_policies_do_not_preview(whole_scheduler, kind):
    scheduler, _ = whole_scheduler
    head, exact = _exact_request(scheduler)
    if kind == "unknown_profile":
        scheduler.memory_monitor.prefill_prediction_components = lambda *a: {"profile": "unknown"}
    else:
        scheduler._prefill_whole_reservation = False
    envelope = scheduler._pipeline_progress_envelope(head)
    assert envelope.context_tokens == PROMPT
    assert envelope.prefill_chunk_tokens == CHUNK
    exact.preview_cached_tokens.assert_not_called()


@pytest.mark.parametrize("actual", [0, CHUNK, HIT])
def test_prepared_actual_result_supersedes_metadata_preview(whole_scheduler, actual):
    scheduler, _ = whole_scheduler
    head, exact = _exact_request(scheduler)
    _prepared(scheduler, head, actual)
    assert scheduler._remaining_exact_prefill_tokens(head) == PROMPT - actual - 1
    envelope = scheduler._pipeline_progress_envelope(head)
    assert envelope.prefill_chunk_tokens == min(CHUNK, PROMPT - actual - 1)
    exact.preview_cached_tokens.assert_not_called()


def test_reconcile_unchanged_quote_keeps_reservation(whole_scheduler):
    scheduler, _ = whole_scheduler
    head, _ = _exact_request(scheduler)
    assert scheduler._reserve_pipeline_memory(head)
    before = scheduler._request_memory_budget.reservations[head.request_id]
    _prepared(scheduler, head, HIT)
    scheduler._request_memory_budget.try_resize = MagicMock(
        side_effect=AssertionError("unchanged quote must retain its ownership"))
    assert scheduler._reconcile_prefix_workspace_budget(head)
    assert scheduler._request_memory_budget.reservations[head.request_id] is before


@pytest.mark.parametrize("free_gib, allowed", [(2, False), (16, True)])
def test_actual_miss_reprices_atomically_without_losing_other_work(
        whole_scheduler, free_gib, allowed):
    scheduler, physical = whole_scheduler
    head, _ = _exact_request(scheduler)
    physical["free"] = free_gib * GIB
    budget = scheduler._request_memory_budget
    old_decode = RequestMemoryEnvelope(PROMPT, GIB // 8, 0, GIB // 4, GIB // 4)
    budget.reservations["old-decode"] = old_decode
    budget.mark_decoding("old-decode")
    budget.shared_reclaim_bytes = GIB // 8
    scheduler._refresh_pipeline_shared_budget = lambda: None
    assert scheduler._reserve_pipeline_memory(head)
    before = budget.reservations[head.request_id]
    shared_and_decode = budget.reserved_bytes() - before.total_bytes
    assert shared_and_decode == old_decode.decode_bytes + GIB // 8
    _prepared(scheduler, head, 0)
    budget.release = MagicMock(side_effect=AssertionError("atomic reprice must not release first"))
    assert scheduler._reconcile_prefix_workspace_budget(head) is allowed
    after = budget.reservations[head.request_id]
    _assert_context_and_decode_unchanged(before, after)
    assert budget.reservations["old-decode"] is old_decode
    assert budget.decoding == {"old-decode"}
    assert budget.reserved_bytes() - after.total_bytes == shared_and_decode
    if allowed:
        assert after.prefill_chunk_tokens == CHUNK
        assert after.prefill_bytes > before.prefill_bytes
    else:
        assert after is before


def _deferred_restore(scheduler, monkeypatch, kind):
    head, exact = _exact_request(scheduler)
    scheduler.waiting.append(head)
    scheduler.batch_generator = MagicMock()
    scheduler._ensure_batch_generator = MagicMock()
    scheduler._do_external_prefill = MagicMock(side_effect=AssertionError("unpriced forward"))
    scheduler._begin_prefill = MagicMock(side_effect=AssertionError("unpriced chunked forward"))
    scheduler.batch_generator.insert.side_effect = AssertionError("unpriced generation kickoff")
    monkeypatch.setattr(scheduler_module.mx, "stream", lambda *a, **kw: nullcontext())
    actual = CHUNK if kind == "shorter" else (HIT if kind == "invalid" else 0)

    def prepare(req):
        assert req.request_id in scheduler._request_memory_budget.reservations
        _prepared(scheduler, req, actual)

    scheduler._prepare_prefix_cache_for_request = MagicMock(side_effect=prepare)
    scheduler._validate_cache = MagicMock(return_value=kind != "invalid")
    return head, exact


@pytest.mark.parametrize("kind", ["miss", "shorter", "invalid"])
def test_restore_race_defers_without_forward_or_repeated_optimistic_preview(
        whole_scheduler, monkeypatch, kind):
    scheduler, physical = whole_scheduler
    physical["free"] = 2 * GIB
    head, exact = _deferred_restore(scheduler, monkeypatch, kind)
    now = {"value": 100.0}
    monkeypatch.setattr(scheduler_module.time, "monotonic", lambda: now["value"])

    scheduled, rejected = scheduler._schedule_waiting()
    assert scheduled == rejected == []
    assert list(scheduler.waiting) == [head]
    assert head.memory_wait_reason == "context_budget"
    assert head.request_id in scheduler._prefix_cache_prepared
    assert head.request_id not in scheduler._request_memory_budget.reservations
    if kind != "shorter":
        assert head.cached_tokens == 0
        assert head.prompt_cache is None
        assert head.remaining_tokens == head.prompt_token_ids
    preview_calls = exact.preview_cached_tokens.call_count
    assert preview_calls == 1
    assert scheduler._prepare_prefix_cache_for_request.call_count == 1

    now["value"] = 101.0
    assert scheduler._schedule_waiting() == ([], [])
    assert exact.preview_cached_tokens.call_count == preview_calls
    assert scheduler._prepare_prefix_cache_for_request.call_count == 1

    now["value"] = 162.0
    scheduled, rejected = scheduler._schedule_waiting()
    assert scheduled == []
    assert [item.error_code for item in rejected] == ["memory_admission_stalled"]
    assert not scheduler.waiting
    assert head.request_id not in scheduler.requests
    assert head.request_id not in scheduler._prefix_cache_prepared
    assert head.request_id not in scheduler._request_memory_budget.reservations
    scheduler._do_external_prefill.assert_not_called()
    scheduler._begin_prefill.assert_not_called()
    scheduler.batch_generator.insert.assert_not_called()


def test_cancelling_repriced_waiter_releases_all_admission_state(whole_scheduler, monkeypatch):
    scheduler, physical = whole_scheduler
    physical["free"] = 2 * GIB
    head, _ = _deferred_restore(scheduler, monkeypatch, "shorter")
    assert scheduler._schedule_waiting() == ([], [])
    scheduler._do_abort_request(head.request_id, allow_cache_store=False)
    assert not scheduler.waiting
    assert head.request_id not in scheduler.requests
    assert head.request_id not in scheduler._prefix_cache_prepared
    assert head.request_id not in scheduler._request_memory_budget.reservations
