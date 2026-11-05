"""Production singleton attention preserves the historical native decode path."""
import importlib.util
import os
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_map

from test_deepseek_v41 import tiny
from omlx.patches.deepseek_v41 import language
from omlx.patches.deepseek_v41.cache import DeepseekV41Cache
from omlx.patches.deepseek_v41.head import project_logits
from omlx.patches.deepseek_v41.quantization import QuantizedProjection, pack_activation


def _historical_oracle():
    path = os.environ.get("DEEPSEEK_SINGLETON_ORACLE")
    if path is None:
        pytest.skip("Set DEEPSEEK_SINGLETON_ORACLE to the independently built historical extension")
    path = Path(path)
    assert path.is_file(), path
    spec = importlib.util.spec_from_file_location("legacy_singleton_oracle", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.forward


@pytest.mark.skipif(mx.default_device() == mx.cpu, reason="Production native ABI control requires GPU")
@pytest.mark.parametrize("native_capability", [True, False], ids=["historical-native", "missing-capability"])
def test_singleton_64_head_512_dim_with_packed_lsq_head(monkeypatch, native_capability):
    historical = _historical_oracle() if native_capability else None
    has_symbol = language.attention_fast.has_symbol
    if native_capability:
        assert has_symbol("packed_sparse_attention_single_decode")
    previous_limit = mx.set_memory_limit(1024**3)
    previous_cache = mx.set_cache_limit(0)
    try:
        mx.random.seed(741)
        config = tiny(dim=5120, n_heads=64, head_dim=512, rope_head_dim=64,
            q_lora_rank=32, o_groups=8, o_lora_rank=128, window_size=256)
        attention = language.Attention(config, 0)
        attention.update(tree_map(lambda x: x.astype(mx.bfloat16), attention.parameters()))
        attention.attn_sink = mx.zeros((64,), mx.float32)
        weight, scales, biases = mx.quantize(mx.random.normal((128, 5120)).astype(mx.bfloat16),
            group_size=64, bits=8)
        head = QuantizedProjection(mx.tile(weight, (1010, 1)), mx.tile(scales, (1010, 1)),
            8, "affine", biases=mx.tile(biases, (1010, 1)), group_size=64, quantize_input=False)
        assert head.weight.shape == (129280, 1280)
        mx.eval(attention.parameters(), head.weight, head.scales, head.biases)
        monkeypatch.setattr(language, "DS41_NATIVE_DECODE", True)
        monkeypatch.setattr(language.attention_fast, "has_symbol", lambda name:
            False if name == "packed_sparse_attention_single_decode" and not native_capability
            else has_symbol(name))
        def rejected_multi(*args, **kwargs):
            raise AssertionError("Singleton must not enter the stable multi-query native ABI")
        monkeypatch.setattr(language.attention_fast, "packed_sparse_attention", rejected_multi)
        native = language.attention_fast.packed_sparse_attention_single_decode
        native_shapes = []
        def observed_native(*args, **kwargs):
            native_shapes.append((args[0].shape, args[1].shape, args[2].shape, args[3].shape))
            return native(*args, **kwargs)
        monkeypatch.setattr(language.attention_fast, "packed_sparse_attention_single_decode", observed_native)
        fallback_shapes = []
        fallback = language.packed_sparse_attention
        def observed_fallback(q, window, pooled, wi, ci, sink, scale, **kwargs):
            fallback_shapes.append((q.shape, window.shape, pooled.shape, ci.shape))
            return fallback(q, window, pooled, wi, ci, sink, scale, **kwargs)
        monkeypatch.setattr(language, "packed_sparse_attention", observed_fallback)
        x = mx.random.normal((1, 1, 5120)).astype(mx.bfloat16)
        for history in (0, 256):
            cached = pack_activation(mx.random.normal((1, history, 512)).astype(mx.bfloat16))
            actual_cache = DeepseekV41Cache(0)
            expected_cache = DeepseekV41Cache(0)
            actual_cache[1] = cached
            expected_cache[1] = cached
            actual = attention(x, actual_cache, {}, history)
            if native_capability:
                monkeypatch.setattr(language.attention_fast, "packed_sparse_attention_single_decode", historical)
            else:
                monkeypatch.setattr(language, "DS41_NATIVE_DECODE", False)
            expected = attention(x, expected_cache, {}, history)
            monkeypatch.setattr(language, "DS41_NATIVE_DECODE", True)
            monkeypatch.setattr(language.attention_fast, "packed_sparse_attention_single_decode", observed_native)
            np.testing.assert_array_equal(actual.view(mx.uint16), expected.view(mx.uint16))
            np.testing.assert_array_equal(actual_cache[1], expected_cache[1])
            logits = project_logits(actual, head)
            expected_logits = project_logits(expected, head)
            np.testing.assert_array_equal(logits, expected_logits)
            assert logits.shape == (1, 1, 129280) and logits.dtype == mx.float32
            assert bool(mx.all(mx.isfinite(logits)).item())
        if native_capability:
            assert len(native_shapes) == 2 and not fallback_shapes
            assert all(shape[0] == (1, 64, 1, 512) for shape in native_shapes)
            assert native_shapes[-1][1] == (1, 1, 257, 528)
        else:
            assert not native_shapes and len(fallback_shapes) == 4
            assert all(shape[0] == (1, 1, 64, 512) for shape in fallback_shapes)
            assert fallback_shapes[-1][1] == (1, 257, 528)
    finally:
        mx.set_cache_limit(previous_cache)
        mx.clear_cache()
        mx.set_memory_limit(previous_limit)
