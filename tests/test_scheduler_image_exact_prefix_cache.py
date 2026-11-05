"""Image identity, checkpoint recording and restored embedding offsets."""

import hashlib

import mlx.core as mx
import pytest

from omlx.patches.deepseek_v41.prefix_cache import CHUNK
from test_scheduler_exact_prefix_cache import _request, _scheduler


def _hash(value):
    return hashlib.sha256(value.encode()).hexdigest()


def image_request(name, *, first="red", second="blue", appended=False):
    request = _request(name, CHUNK * 3 + 9)
    positions = [CHUNK + 10, CHUNK * 2 + 10]
    identities = [_hash("vlm-program"), _hash("vlm-program:" + first),
                  _hash("vlm-program:" + first + ":" + second)]
    request.vlm_cache_key_ranges = [(0, identities[0])]
    for position, identity in zip(positions, identities[1:]):
        request.prompt_token_ids[position:position + 194] = [129264] * 194
        request.vlm_cache_key_ranges.append((position, identity))
    if appended:
        position = CHUNK * 3 + 2
        request.prompt_token_ids[position:position + 3] = [129264] * 3
        request.vlm_cache_key_ranges.append((position, _hash("third-image")))
    request.vlm_cache_key_start = 0
    request.vlm_image_hash = request.vlm_cache_key_ranges[-1][1]
    request.vlm_inputs_embeds = mx.arange(request.num_prompt_tokens)[None, :, None]
    return request


def prefill_image(scheduler, request):
    return scheduler._do_external_prefill(
        request, request.remaining_tokens, request.prompt_cache,
        (request.vlm_inputs_embeds, {}, request.cached_tokens),
    )


def test_image_cold_records_and_warm_restores_embeddings(mock_tokenizer, monkeypatch):
    scheduler = _scheduler(mock_tokenizer, monkeypatch)
    cold = image_request("cold")
    scheduler._prepare_prefix_cache_for_request(cold)
    prefill_image(scheduler, cold)
    assert scheduler.model._omlx_exact_prefix_cache.stores == 3

    warm = image_request("warm")
    scheduler._prepare_prefix_cache_for_request(warm)
    assert warm.cached_tokens == warm.api_cached_tokens == 3 * CHUNK
    seen = []

    def forward(tokens, cache=None, inputs_embeds=None, **kwargs):
        seen.extend(inputs_embeds[0, :, 0].tolist())
        cache[0].cache[0] = cache[0].cache[0] + tokens.shape[1]
        return mx.zeros((1, tokens.shape[1], 8))

    scheduler.model._omlx_prefill = forward
    _, last = prefill_image(scheduler, warm)
    assert seen == list(range(3 * CHUNK, warm.num_prompt_tokens - 1))
    assert last == warm.prompt_token_ids[-1:]


@pytest.mark.parametrize("kwargs,expected", [
    ({"first": "green"}, CHUNK),
    ({"second": "green"}, CHUNK * 2),
    ({"appended": True}, CHUNK * 3),
])
def test_changed_or_appended_image_preserves_only_valid_prefix(
    mock_tokenizer, monkeypatch, kwargs, expected,
):
    scheduler = _scheduler(mock_tokenizer, monkeypatch)
    cold = image_request("cold")
    scheduler._prepare_prefix_cache_for_request(cold)
    prefill_image(scheduler, cold)
    changed = image_request("changed", **kwargs)
    scheduler._prepare_prefix_cache_for_request(changed)
    assert changed.cached_tokens == changed.api_cached_tokens == expected
    assert changed.remaining_tokens == changed.prompt_token_ids[expected:]


def test_unidentified_image_embeddings_bypass_exact_cache(mock_tokenizer, monkeypatch):
    scheduler = _scheduler(mock_tokenizer, monkeypatch)
    request = image_request("unidentified")
    request.vlm_image_hash = None
    request.vlm_cache_key_ranges = None
    scheduler._prepare_prefix_cache_for_request(request)
    prefill_image(scheduler, request)
    assert request.cached_tokens == 0
    assert scheduler.model._omlx_exact_prefix_cache.stores == 0


def test_invalid_image_identity_cannot_restore_text_state(mock_tokenizer, monkeypatch):
    scheduler = _scheduler(mock_tokenizer, monkeypatch)
    cold = image_request("cold")
    scheduler._prepare_prefix_cache_for_request(cold)
    prefill_image(scheduler, cold)
    bad = image_request("bad")
    bad.vlm_image_hash = "invalid"
    scheduler._prepare_prefix_cache_for_request(bad)
    assert bad.cached_tokens == 0
    assert bad.remaining_tokens == bad.prompt_token_ids


@pytest.mark.parametrize("shortage", [0, 1])
def test_prefill_pool_reclaim_retains_unneeded_checkpoints(
    mock_tokenizer, monkeypatch, shortage,
):
    from omlx.scheduler import Scheduler
    scheduler = _scheduler(mock_tokenizer, monkeypatch)
    cold = image_request("cold")
    scheduler._prepare_prefix_cache_for_request(cold)
    prefill_image(scheduler, cold)
    exact = scheduler.model._omlx_exact_prefix_cache
    original_bytes = exact.bytes
    pooled = [10**8]
    monkeypatch.setattr(Scheduler, "_clear_cache", lambda self: pooled.__setitem__(0, 0))
    monkeypatch.setattr(scheduler, "_current_usage_bytes", lambda: 1000 + exact.bytes + pooled[0])
    target = 1000 + original_bytes + 100 - shortage
    after = scheduler._reclaim_prefill_headroom_for_request(100, target)
    assert after + 100 <= target
    assert len(exact.entries) == 3 - shortage
    assert exact.evictions == shortage


@pytest.mark.asyncio
async def test_engine_pool_passes_headroom_budget_on_owning_executor(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import get_ident
    from types import SimpleNamespace
    import omlx.engine_pool as module

    calls = []
    main_thread = get_ident()

    class BudgetScheduler:
        def _reclaim_prefill_headroom(self):
            raise AssertionError("Unbounded reclaim was selected")

        def _reclaim_prefill_headroom_for_request(self, additional, target):
            calls.append((additional, target, get_ident()))

    pool = module.EnginePool.__new__(module.EnginePool)
    monkeypatch.setattr(module, "get_phys_footprint", lambda: 1000)
    with ThreadPoolExecutor(max_workers=1) as executor:
        core = SimpleNamespace(scheduler=BudgetScheduler(), _mlx_executor=executor)
        pool._entries = {"model": SimpleNamespace(engine=core)}
        await pool._reclaim_pooled_buffers_for_prefill(
            "model", "request", predicted_bytes=123, target_bytes=456,
        )
    assert len(calls) == 1
    assert calls[0][:2] == (123, 456)
    assert calls[0][2] != main_thread
