"""Bounded CPU checks for the exact imports and embedding adapter used at load."""
import importlib
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from omlx.patches.deepseek_v41.quantization import QuantizedEmbedding
from test_deepseek_v41 import write_affine_checkpoint


@pytest.fixture(autouse=True)
def cpu_only():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


def test_transitive_loader_dependencies_are_from_candidate_source():
    package = Path(__file__).resolve().parents[1] / 'omlx/patches/deepseek_v41'
    for name in ('loading', 'convert', 'model', 'processing', 'quantization', 'storage', 'native_metadata'):
        module = importlib.import_module('omlx.patches.deepseek_v41.' + name)
        assert Path(module.__file__).resolve() == package / (name + '.py')


@pytest.mark.parametrize('bits', [2, 3, 4])
def test_quantized_embedding_only_dequantizes_requested_rows(monkeypatch, bits):
    from omlx.patches.deepseek_v41.quantization import QuantizedProjection
    weight = mx.array(np.random.default_rng(41).normal(size=(16, 64)).astype('float32')).astype(mx.bfloat16)
    packed, scales, biases = mx.quantize(weight, group_size=64, bits=bits)
    embedding = QuantizedEmbedding(packed, scales, bits, 'affine', biases=biases,
                                  group_size=64, quantize_input=False)
    indices = mx.array([[3, 7, 3], [2, 0, 15]], mx.int32)
    expected = mx.dequantize(packed[indices], scales[indices], biases[indices], group_size=64, bits=bits)
    original = mx.dequantize
    shapes = []
    def observed(value, *args, **kwargs):
        shapes.append(value.shape)
        return original(value, *args, **kwargs)
    monkeypatch.setattr(mx, 'dequantize', observed)
    actual = embedding(indices)
    np.testing.assert_array_equal(actual.view(mx.uint16), expected.view(mx.uint16))
    assert shapes == [packed[indices].shape]
    assert actual.shape == (2, 3, 64)
    assert isinstance(embedding, QuantizedProjection)


def test_embedding_rejects_activation_quantization():
    packed, scales, biases = mx.quantize(mx.ones((8, 64), mx.bfloat16), group_size=64, bits=3)
    embedding = QuantizedEmbedding(packed, scales, 3, 'affine', biases=biases,
                                  group_size=64, quantize_input=True)
    with pytest.raises(ValueError, match='without activation quantization'):
        embedding(mx.array([1], mx.int32))


def test_tiny_affine_checkpoint_loads_embedding_adapter(tmp_path):
    from omlx.patches.deepseek_v41.loading import load
    import json
    from omlx.patches.deepseek_v41.convert import convert
    source, _ = write_affine_checkpoint(tmp_path, vision=False, bits=3)
    target = tmp_path / "converted"
    convert(source, target)
    index_path = target / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    key = "language_model.embed"
    shard_name = index["weight_map"][key + ".weight"]
    shard = target / shard_name
    values = dict(mx.load(str(shard)))
    packed, scales, biases = mx.quantize(values[key + ".weight"].astype(mx.bfloat16), group_size=64, bits=3)
    values.update({key + ".weight": packed, key + ".scales": scales, key + ".biases": biases})
    mx.eval(list(values.values()))
    mx.save_safetensors(str(shard), values)
    index["weight_map"].update({key + ".scales": shard_name, key + ".biases": shard_name})
    index_path.write_text(json.dumps(index))
    config_path = target / "config.json"
    config = json.loads(config_path.read_text())
    config["omlx_deepseek_v41"]["quantized_modules"][key] = {
        "bits": 3, "mode": "affine", "group_size": 64, "quantize_input": False}
    config_path.write_text(json.dumps(config))
    model, _ = load(target)
    try:
        embedding = model.language_model.embed
        assert isinstance(embedding, QuantizedEmbedding)
        indices = mx.array([[1, 3, 1]], mx.int32)
        actual = embedding(indices)
        expected = mx.dequantize(embedding.weight[indices], embedding.scales[indices],
                                 embedding.biases[indices], group_size=embedding.group_size, bits=embedding.bits)
        np.testing.assert_array_equal(actual.view(mx.uint16), expected.view(mx.uint16))
    finally:
        model.close()
