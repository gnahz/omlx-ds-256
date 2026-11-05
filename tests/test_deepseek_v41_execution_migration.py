"""Production dispatch policies are bounded, reversible and numerically neutral."""
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from test_deepseek_v41 import load_reference_weights, tiny
from omlx.patches.deepseek_v41 import language
from omlx.patches.mlx_lm_mtp import batch_generator as bg, copy_draft


def test_prefill_pipeline_drains_before_index_and_restores_cache_limit(monkeypatch):
    events = []
    monkeypatch.setattr(language, "DS41_PREFILL_POOL", 3*1024**3)
    monkeypatch.setattr(language, "DS41_PREFILL_PIPELINE", True)
    monkeypatch.setattr(mx, "set_cache_limit", lambda n: events.append(("limit", n)) or 99)
    monkeypatch.setattr(mx, "async_eval", lambda *a: events.append(("async", a)))
    monkeypatch.setattr(mx, "eval", lambda *a: events.append(("eval", a)))
    monkeypatch.setattr(mx, "clear_cache", lambda: events.append(("clear",)))
    with language._LayerExecution(True) as execution:
        execution.submit("h0", "p0", 0)
        execution.before_layer(1, SimpleNamespace(index_source_layers=(2,)))
        assert events[-1][0] == "async"
        execution.before_layer(2, SimpleNamespace(index_source_layers=(2,)))
        assert events[-1] == ("eval", ("h0", "p0"))
        execution.submit("h2", "p2", 2)
    assert events == [("limit", 3*1024**3), ("async", ("h0", "p0")),
        ("eval", ("h0", "p0")), ("async", ("h2", "p2")),
        ("eval", ("h2", "p2")), ("limit", 99), ("clear",)]


def test_prefill_exception_drains_submissions_and_restores_pool(monkeypatch):
    events = []
    monkeypatch.setattr(language, "DS41_PREFILL_POOL", 3*1024**3)
    monkeypatch.setattr(language, "DS41_PREFILL_PIPELINE", True)
    monkeypatch.setattr(mx, "set_cache_limit", lambda n: events.append(("limit", n)) or 77)
    monkeypatch.setattr(mx, "async_eval", lambda *a: events.append(("async",)))
    monkeypatch.setattr(mx, "eval", lambda *a: events.append(("drain",)))
    monkeypatch.setattr(mx, "synchronize", lambda: events.append(("sync",)))
    monkeypatch.setattr(mx, "clear_cache", lambda: events.append(("clear",)))
    with pytest.raises(RuntimeError, match="layer failed"):
        with language._LayerExecution(True) as execution:
            execution.submit("h", "p", 0)
            raise RuntimeError("layer failed")
    assert events[-4:] == [("drain",), ("sync",), ("limit", 77), ("clear",)]


def test_periodic_decode_async_keeps_four_layer_default_and_can_disable(monkeypatch):
    events = []
    monkeypatch.setattr(language, "DS41_DECODE_ASYNC", 4)
    monkeypatch.setattr(mx, "async_eval", lambda *a: events.append(a))
    with language._LayerExecution(False) as execution:
        for i in range(10): execution.submit(i, -i, i)
    assert events == [(3, -3), (7, -7)]
    monkeypatch.setattr(language, "DS41_DECODE_ASYNC", 0)
    with language._LayerExecution(False) as execution:
        execution.submit(11, -11, 11)
    assert len(events) == 2


def test_dspark_uses_its_existing_copy_index_without_generic_context_copy(monkeypatch):
    host = object()
    monkeypatch.setattr(bg, "_dspark_host", lambda model: host)
    monkeypatch.setattr(bg._context_copy, "ContextCopy", lambda **kw:
        (_ for _ in ()).throw(AssertionError("Generic copier must not initialize for DSpark")))
    assert bg._context_copy_drafts(SimpleNamespace(model=object()), SimpleNamespace(), [7]) == []
    assert copy_draft.ENABLED
    assert (copy_draft.NGRAM, copy_draft.MIN_DRAFT, copy_draft.MAX_DRAFT, copy_draft.MAX_MATCH) == (4, 4, 4, 16)
    index = copy_draft.CopyIndex([1, 2, 3, 4, 5, 6, 7, 8]*5)
    assert index.propose(4) == [1, 2, 3, 4]


def test_generic_context_copy_remains_available_to_other_models(monkeypatch):
    monkeypatch.setattr(bg, "_dspark_host", lambda model: None)
    calls = []
    copier = SimpleNamespace(extend=lambda *a: calls.append(a) or False,
                             propose=lambda room: [9, room])
    batch = SimpleNamespace(model=object(), tokens=[[1, 2]], max_tokens=[20], _num_tokens=[3])
    assert bg._context_copy_drafts(batch, SimpleNamespace(context_copy=copier), [4]) == [9, 15]
    assert calls == [([1, 2], [4])]


@pytest.mark.parametrize("lengths,expected", [((4,)*8, [(4,)*8]), ((5,)*8, [(5,)*6, (5,)*2])])
def test_copy_index_verify_cohorts_keep_all_rows_and_order(lengths, expected, monkeypatch):
    from omlx.patches.deepseek_v41.mtp import DSparkMixin
    from omlx.patches.deepseek_v41 import batched_verify
    calls = []
    def forward(host, ids, caches):
        calls.append(tuple(value.shape[1] for value in ids))
        return list(caches)
    monkeypatch.setattr(batched_verify, "forward", forward)
    inputs = [mx.zeros((1, n), mx.int32) for n in lengths]
    host = SimpleNamespace(pipeline_size=1, pipeline_layers=[])
    assert DSparkMixin.mtp_verify_requests(host, inputs, list(range(8))) == list(range(8))
    assert calls == expected


def test_invalid_ragged_cohort_is_rejected_before_any_cache_changes(monkeypatch):
    from omlx.patches.deepseek_v41.mtp import DSparkMixin
    from omlx.patches.deepseek_v41 import batched_verify
    monkeypatch.setattr(batched_verify, "forward", lambda *a:
        (_ for _ in ()).throw(AssertionError("Invalid cohort must fail before forwarding")))
    host = SimpleNamespace(pipeline_size=1, pipeline_layers=[])
    with pytest.raises(ValueError, match="unpadded rows"):
        DSparkMixin.mtp_verify_requests(host, [mx.zeros((1, 5), mx.int32), mx.zeros((1, 9), mx.int32)], [0, 1])


@pytest.mark.parametrize("pipeline", [False, True])
def test_prefill_pool_pipeline_preserves_full_logits_and_cache(pipeline, monkeypatch):
    model = language.LanguageModel(tiny(max_seq_len=512))
    load_reference_weights(model)
    tokens = mx.array([[3+i%20 for i in range(256)]])
    expected_cache, actual_cache = model.make_cache(), model.make_cache()
    monkeypatch.setattr(language, "DS41_PREFILL_POOL", 0)
    monkeypatch.setattr(language, "DS41_PREFILL_PIPELINE", False)
    expected = model(tokens, cache=expected_cache)
    mx.eval(expected)
    monkeypatch.setattr(language, "DS41_PREFILL_POOL", 3*1024**3)
    monkeypatch.setattr(language, "DS41_PREFILL_PIPELINE", pipeline)
    actual = model(tokens, cache=actual_cache)
    np.testing.assert_array_equal(actual.view(mx.uint32), expected.view(mx.uint32))
    for a, b in zip(actual_cache, expected_cache):
        for x, y in zip(a.cache, b.cache):
            if x is not None: np.testing.assert_array_equal(x.view(mx.uint8), y.view(mx.uint8))
