"""Production LSQ vocabulary heads retain packed storage and FP32 arithmetic."""
import json
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from omlx.patches.deepseek_v41 import dspark
from omlx.patches.deepseek_v41.head import project_logits
from omlx.patches.deepseek_v41.quantization import QuantizedProjection
from test_deepseek_v41 import write_checkpoint


def packed_head(width=128, rows=128):
    mx.random.seed(741)
    packed, scales, biases = mx.quantize(
        mx.random.normal((rows, width)).astype(mx.bfloat16), group_size=64, bits=8)
    return QuantizedProjection(packed, scales, 8, "affine", biases=biases,
                               group_size=64, quantize_input=False)


def original_logits(x, head):
    """Historical production projection: packed affine weights, FP32 operands."""
    return mx.quantized_matmul(x.astype(mx.float32), head.weight,
        head.scales.astype(mx.float32), head.biases.astype(mx.float32),
        transpose=True, group_size=head.group_size, bits=head.bits, mode=head.mode)


@pytest.mark.parametrize("rows", [1, 2, 5, 8, 11, 12, 128])
@pytest.mark.parametrize("input_width", [128, 5120])
def test_quantized_head_uses_original_arithmetic_without_expansion(monkeypatch, rows, input_width):
    head = packed_head(width=input_width)
    x = mx.random.normal((1, rows, input_width)).astype(mx.bfloat16)
    expected = original_logits(x, head)
    mx.eval(expected, head.weight, head.scales, head.biases)
    packed = head.weight
    def forbidden(*args, **kwargs):
        raise AssertionError("Vocabulary weights/activations must stay in their original format")
    monkeypatch.setattr(mx, "dequantize", forbidden)
    monkeypatch.setattr(mx, "quantize", forbidden)
    actual = project_logits(x, head)
    assert actual.shape == (1, rows, 128)
    assert actual.dtype == mx.float32
    np.testing.assert_array_equal(actual.view(mx.uint32), expected.view(mx.uint32))
    assert head.weight is packed
    # This small-width QMV fixture has serial equality through 11 rows.
    # Production-width reductions depend on row count; retain the original
    # same-shaped arithmetic rather than changing its reduction order.
    if input_width == 128 and rows <= 11:
        serial = original_logits(x[:, -1:], head)
        np.testing.assert_array_equal(actual[:, -1:].view(mx.uint32), serial.view(mx.uint32))


def test_packed_head_requires_metadata_and_unquantized_activations():
    head = packed_head()
    x = mx.ones((1, 1, 128), mx.bfloat16)
    with pytest.raises(ValueError, match="require projection metadata"):
        project_logits(x, head.weight)
    head.quantize_input = True
    with pytest.raises(ValueError, match="must not quantize activations"):
        project_logits(x, head)


@pytest.mark.parametrize("width", [1, 2, 5])
def test_draft_and_batched_heads_dispatch_the_packed_module(width):
    model = SimpleNamespace(head=packed_head())
    xs = [mx.random.normal((1, width, 128)).astype(mx.bfloat16) for _ in range(2)]
    for x in xs:
        actual = dspark.draft_logits(model, x)
        np.testing.assert_array_equal(actual.view(mx.uint32), original_logits(x, model.head).view(mx.uint32))
    actual = dspark.batch_logits(model, xs, width, [False, False])
    expected = original_logits(mx.concatenate(xs, axis=1), model.head)
    for index, value in enumerate(actual):
        np.testing.assert_array_equal(value.view(mx.uint32), expected[:, index*width:(index+1)*width].view(mx.uint32))


def test_converted_checkpoint_forward_with_packed_vocabulary(tmp_path):
    from omlx.patches.deepseek_v41.convert import convert
    from omlx.patches.deepseek_v41.loading import load
    source, _ = write_checkpoint(tmp_path, vision=False, dim=64, moe_inter_dim=64,
                                 max_seq_len=256)
    target = tmp_path / "converted"
    convert(source, target)
    index_path = target / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    key = "language_model.head"
    shard_name = index["weight_map"][key + ".weight"]
    shard = target / shard_name
    values = dict(mx.load(str(shard)))
    packed, scales, biases = mx.quantize(values[key + ".weight"].astype(mx.bfloat16),
                                         group_size=64, bits=8)
    values.update({key + ".weight": packed, key + ".scales": scales, key + ".biases": biases})
    mx.eval(list(values.values()))
    mx.save_safetensors(str(shard), values)
    index["weight_map"].update({key + ".scales": shard_name, key + ".biases": shard_name})
    index_path.write_text(json.dumps(index))
    config_path = target / "config.json"
    config = json.loads(config_path.read_text())
    config["omlx_deepseek_v41"]["quantized_modules"][key] = {
        "bits": 8, "mode": "affine", "group_size": 64, "quantize_input": False}
    config_path.write_text(json.dumps(config))
    model, _ = load(target)
    try:
        language = model.language_model
        assert isinstance(language.head, QuantizedProjection)
        tokens = mx.array([[3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]], mx.int32)
        hidden = language(tokens, cache=language.make_cache(), skip_logits=True)
        actual = language(tokens, cache=language.make_cache())
        expected = original_logits(hidden, language.head)
        assert actual.shape == (1, 12, 64)
        assert actual.dtype == mx.float32
        np.testing.assert_array_equal(actual.view(mx.uint32), expected.view(mx.uint32))
    finally:
        model.close()
