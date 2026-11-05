"""Exact-prefix ownership, memory protection, and request usage integration."""

import hashlib
import struct
from types import SimpleNamespace

import mlx.core as mx
import pytest

from omlx.patches.deepseek_v41.prefix_cache import CHUNK, ExactPrefixCache
from omlx.request import Request, SamplingParams
from omlx.scheduler import Scheduler, SchedulerConfig, _AdmissionEstimate
from test_deepseek_v41_prefix_cache import sample


class _Model:
    def __init__(self):
        self.layers = [SimpleNamespace()]
        self.args = SimpleNamespace(num_hidden_layers=1)
        self._omlx_exact_prefix_cache = ExactPrefixCache(2**20)

    def make_cache(self):
        return sample(0)

    def parameters(self):
        return {}

    def __call__(self, tokens, cache=None, **kwargs):
        cache[0].cache[0] = cache[0].cache[0] + tokens.shape[1]
        return mx.zeros((1, tokens.shape[1], 8))


def _scheduler(mock_tokenizer, monkeypatch):
    scheduler = Scheduler(
        model=_Model(), tokenizer=mock_tokenizer,
        config=SchedulerConfig(prefill_step_size=CHUNK, paged_cache_block_size=0),
    )
    scheduler._prefill_memory_guard = True
    scheduler._memory_limit_bytes = 2**50
    scheduler._memory_hard_limit_bytes = 2**50
    scheduler._memory_abort_limit_bytes = 2**50
    monkeypatch.setattr(scheduler, "_current_usage_bytes", lambda: 0)
    return scheduler


def _request(name="r", count=CHUNK * 2 + 1):
    tokens = list(range(count))
    request = Request(request_id=name, prompt=tokens, sampling_params=SamplingParams())
    request.prompt_token_ids = tokens
    request.num_prompt_tokens = count
    request.benchmark_trace = True
    return request


def _prefill(scheduler, request, chunked):
    if chunked:
        state = scheduler._begin_prefill(
            request, request.remaining_tokens, request.prompt_cache
        )
        while not scheduler._step_prefill_chunk(state):
            pass
    else:
        scheduler._do_external_prefill(
            request, request.remaining_tokens, request.prompt_cache
        )


@pytest.mark.parametrize("chunked", [False, True])
def test_exact_cold_warm_with_memory_guard_enabled(mock_tokenizer, monkeypatch, chunked):
    scheduler = _scheduler(mock_tokenizer, monkeypatch)
    cold = _request()
    cold.benchmark_trace = False
    scheduler._prepare_prefix_cache_for_request(cold)
    assert cold.benchmark_trace
    assert cold.cached_tokens == cold.api_cached_tokens == 0
    _prefill(scheduler, cold, chunked)
    exact = scheduler.model._omlx_exact_prefix_cache
    assert exact.stores == 2
    assert cold.benchmark_prefill_chunks == [CHUNK, CHUNK]
    warm = _request("warm")
    scheduler._prepare_prefix_cache_for_request(warm)
    assert warm.cached_tokens == warm.api_cached_tokens == CHUNK * 2
    assert warm.remaining_tokens == [CHUNK * 2]
    assert scheduler._prefill_memory_guard


@pytest.mark.parametrize("chunked", [False, True])
def test_resized_chunk_invalidates_only_later_checkpoints(mock_tokenizer, monkeypatch, chunked):
    scheduler = _scheduler(mock_tokenizer, monkeypatch)
    request = _request(count=CHUNK * 3 + 1)
    scheduler._prepare_prefix_cache_for_request(request)
    calls = 0

    def shrink_after_first(n, **kwargs):
        nonlocal calls
        calls += 1
        return n if calls == 1 else min(n, CHUNK // 2)

    monkeypatch.setattr(scheduler, "_guard_prefill_chunk", shrink_after_first)
    _prefill(scheduler, request, chunked)
    exact = scheduler.model._omlx_exact_prefix_cache
    assert exact.stores == 1
    assert not request._ds41_prefix_plan.valid
    assert exact.prepare(request.prompt_token_ids)[1] == CHUNK


def test_snapshot_budget_evicts_real_bytes_without_double_counting(mock_tokenizer, monkeypatch):
    scheduler = _scheduler(mock_tokenizer, monkeypatch)
    exact = scheduler._configure_exact_prefix_cache()
    tokens = _request().prompt_token_ids
    _, _, plan = exact.prepare(tokens)
    exact.record(plan, sample(), 0, CHUNK)
    exact.record(plan, sample(CHUNK * 2), CHUNK, CHUNK)
    retained = exact.bytes
    oldest_size = next(iter(exact.entries.values())).size
    baseline = 1000
    scheduler._memory_hard_limit_bytes = baseline + retained + 100
    scheduler._memory_abort_limit_bytes = 0
    scheduler._prefill_abort_margin = 1.0
    monkeypatch.setattr(scheduler, "_current_usage_bytes", lambda: baseline + exact.bytes)
    monkeypatch.setattr(Scheduler, "_clear_cache", lambda self: None)
    # Isolate snapshot accounting from upstream admission safety margins.
    monkeypatch.setattr(scheduler, "_admission_limit_bytes", lambda: scheduler._memory_hard_limit_bytes)
    # Already-resident bytes are measured once, so a fitting copy keeps both entries.
    assert exact.ensure_headroom(100)
    assert len(exact.entries) == 2
    assert exact.ensure_headroom(oldest_size)
    assert len(exact.entries) == 1
    assert exact.evictions == 1
    # Even evicting everything cannot fit a copy larger than the process cap.
    assert not exact.ensure_headroom(scheduler._memory_hard_limit_bytes + 1)
    assert exact.bytes == 0
    assert scheduler._prefill_memory_guard


def test_restore_rejected_by_guard_is_cold_and_reports_zero(mock_tokenizer, monkeypatch):
    scheduler = _scheduler(mock_tokenizer, monkeypatch)
    exact = scheduler._configure_exact_prefix_cache()
    request = _request()
    _, _, plan = exact.prepare(request.prompt_token_ids)
    exact.record(plan, sample(), 0, CHUNK)
    scheduler._memory_hard_limit_bytes = exact.bytes
    scheduler._memory_abort_limit_bytes = 0
    scheduler._prefill_abort_margin = 1
    monkeypatch.setattr(scheduler, "_current_usage_bytes", lambda: exact.bytes)
    monkeypatch.setattr(Scheduler, "_clear_cache", lambda self: None)
    scheduler._prepare_prefix_cache_for_request(request)
    assert request.prompt_cache is None
    assert request.cached_tokens == request.api_cached_tokens == 0
    assert request.remaining_tokens == request.prompt_token_ids
    assert exact.skipped_memory == 1


def test_unknown_guard_limit_refuses_snapshot_allocations(mock_tokenizer, monkeypatch):
    scheduler = _scheduler(mock_tokenizer, monkeypatch)
    scheduler._memory_hard_limit_bytes = 0
    scheduler._memory_abort_limit_bytes = 0
    exact = scheduler._configure_exact_prefix_cache()
    assert not exact.ensure_headroom(1)


def test_admission_evicts_idle_snapshots_before_rejecting(mock_tokenizer, monkeypatch):
    scheduler = _scheduler(mock_tokenizer, monkeypatch)
    exact = scheduler._configure_exact_prefix_cache()
    request = _request()
    _, _, plan = exact.prepare(request.prompt_token_ids)
    exact.record(plan, sample(), 0, CHUNK)
    scheduler._memory_hard_limit_bytes = 2000
    scheduler._memory_abort_limit_bytes = 0
    scheduler._prefill_abort_margin = 1.0
    monkeypatch.setattr(scheduler, "_current_usage_bytes", lambda: 1000 + exact.bytes)
    monkeypatch.setattr(Scheduler, "_clear_cache", lambda self: None)
    monkeypatch.setattr(
        scheduler, "_admission_estimate",
        lambda **kw: _AdmissionEstimate(100, 200, CHUNK, CHUNK, kw["current"] + 300),
    )
    assert scheduler._preflight_memory_check(request) is None
    assert exact.bytes == 0
    assert request.prefill_eviction_retries == 0


@pytest.mark.parametrize("reclaim", ["_reclaim_prefill_headroom", "reset", "_recover_from_cache_error"])
def test_pressure_or_explicit_reset_releases_checkpoints(mock_tokenizer, monkeypatch, reclaim):
    scheduler = _scheduler(mock_tokenizer, monkeypatch)
    exact = scheduler._configure_exact_prefix_cache()
    _, _, plan = exact.prepare(_request().prompt_token_ids)
    exact.record(plan, sample(), 0, CHUNK)
    assert exact.bytes > 0
    scheduler._pending_pressure_clear = True
    getattr(scheduler, reclaim)()
    assert exact.bytes == 0


def test_reprefill_discards_actual_reuse_count(mock_tokenizer, monkeypatch):
    scheduler = _scheduler(mock_tokenizer, monkeypatch)
    request = _request()
    request.cached_tokens = CHUNK * 2
    request.cache_reused_tokens = CHUNK
    scheduler._reset_request_for_reprefill(request)
    assert request.cached_tokens == request.api_cached_tokens == 0
    assert request.remaining_tokens == request.prompt_token_ids


def test_api_output_and_completed_hash_use_full_request_and_actual_reuse(mock_tokenizer, monkeypatch):
    scheduler = _scheduler(mock_tokenizer, monkeypatch)
    request = _request()
    request.cached_tokens = CHUNK * 2  # Includes local resume progress.
    request.cache_reused_tokens = CHUNK
    request.output_token_ids = [50, 51]
    scheduler.requests[request.request_id] = request
    scheduler.running[request.request_id] = request
    scheduler.uid_to_request_id[1] = request.request_id
    response = SimpleNamespace(uid=1, token=52, finish_reason="length", logprobs=None)
    outputs, finished = scheduler._process_batch_responses([response])
    assert request.request_id in finished
    assert outputs[-1].cached_tokens == CHUNK
    assert outputs[-1].completion_tokens == 3
    row = scheduler.exact_prefix_completed[-1]
    assert row["output_tokens"] == 3
    assert row["reused_tokens"] == CHUNK
    assert row["output_token_ids_sha256"] == hashlib.sha256(struct.pack("<3I", 50, 51, 52)).hexdigest()
    for _ in range(20):
        scheduler._record_exact_prefix_completion(request)
    assert len(scheduler.exact_prefix_completed) == 16
