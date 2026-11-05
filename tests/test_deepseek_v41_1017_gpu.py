"""Small-tensor GPU checks against an independent 1017 act_quant oracle.

No checkpoint is loaded. The scale calculation follows the pinned official
kernel's FP32 reciprocal multiply and IEEE exponent/mantissa rounding.
Reference: deepseek-ai/DeepSeek-V4.1-Flash-1017, inference/kernel.py,
revision 1bae0e1088993b9b82de08f59de5be5a34e42170. Its served input is BF16;
FP16/FP32 cases additionally verify the local helper's supported dtypes.
"""

from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from omlx.patches.deepseek_v41 import attn_out, decode_fusions
from omlx.patches.deepseek_v41.language import Attention
from omlx.patches.deepseek_v41.quantization import _quantize_attention_output


@pytest.fixture(autouse=True)
def gpu_device():
    if not mx.metal.is_available():
        pytest.skip("Metal GPU is unavailable")
    previous = mx.default_device()
    mx.set_default_device(mx.gpu)
    yield
    mx.synchronize()
    mx.set_default_device(previous)


def official_inplace_fp8(values):
    """FP32 scale + nearest E4M3FN, ties to even codes, then dequantize."""
    grouped = values.reshape(*values.shape[:-1], -1, 32)
    amax = np.maximum(np.abs(grouped).max(-1, keepdims=True), np.float32(1e-4))
    scale_bits = (amax * np.float32(1 / 448)).view(np.uint32)
    exponents = ((scale_bits >> 23) & 255).astype(np.int32) - 127 + ((scale_bits & 0x7fffff) != 0)
    scales = np.ldexp(np.ones_like(amax), exponents)
    scaled = np.clip(grouped / scales, -448, 448)
    codes = np.arange(127, dtype=np.uint8)
    exponent, fraction = (codes >> 3).astype(np.int32), (codes & 7).astype(np.float32)
    levels = np.where(exponent == 0, fraction * np.float32(2**-9), np.ldexp(1 + fraction / 8, exponent - 7))
    absolute = np.abs(scaled)
    upper = np.clip(np.searchsorted(levels, absolute), 1, 126)
    lower = upper - 1
    dl, du = absolute - levels[lower], levels[upper] - absolute
    selected = np.where((dl < du) | ((dl == du) & ((lower & 1) == 0)), lower, upper)
    quantized = np.copysign(levels[selected], scaled)
    return (quantized * scales).reshape(values.shape).astype(np.float32)


def actual_heads(x):
    holder = SimpleNamespace(_config=SimpleNamespace(attention_output_quantization=True))
    return Attention._output_heads(holder, x)


def expected_for(x):
    # Capture the already-rounded input dtype, then emulate only the official
    # output cast through MLX. FP8 conversion and scale selection are NumPy.
    values = np.asarray(x.astype(mx.float32))
    return mx.array(official_inplace_fp8(values)).astype(x.dtype)


def assert_bits(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    np.testing.assert_array_equal(np.asarray(actual.view(mx.uint8)), np.asarray(expected.view(mx.uint8)))


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16, mx.float32])
@pytest.mark.parametrize("length", [1, 5, 17])
def test_attention_heads_match_official_scale_floor_and_rounding(dtype, length):
    shape = (1, length, 64, 512)
    rng = np.random.default_rng(1017 + length)
    values = rng.normal(size=shape).astype(np.float32)
    # Each row/group sees a broad range, including values below the 1e-4 floor.
    exponents = np.arange(values.size // 32) % 35 - 25
    values = (values.reshape(-1, 32) * np.exp2(exponents[:, None])).reshape(shape)
    x = mx.array(values).astype(dtype)
    result = actual_heads(x)
    assert_bits(result, expected_for(x))
    assert_bits(_quantize_attention_output(x), result)
    grouped = x.reshape(1, length, 8, -1)
    assert_bits(actual_heads(grouped).reshape(shape), result)


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16, mx.float32])
def test_zero_tiny_midpoint_and_power_of_two_boundary_groups(dtype):
    # Repeated full groups control the scale independently of the test value.
    maxima = np.array([0, 1e-20, 1e-7, 1e-4, 448 * 2**-10, 448, 896], np.float32)
    groups = np.zeros((len(maxima), 32), np.float32)
    groups[:, 0] = maxima
    groups[:, 1] = -maxima
    groups[:, 2] = np.float32(-0.0)
    groups[:, 3:9] = np.array([1.0625, -1.0625, 1.1875, -1.1875, 0.0009765625, -0.0009765625])
    groups[:, 3:9] *= np.maximum(maxima[:, None] / 448, np.float32(2**-22))
    # Adjacent FP32 amax values expose ceil(log2()) approximations around 448.
    boundaries = np.zeros((3, 32), np.float32)
    boundaries[:, 0] = [np.nextafter(np.float32(448), np.float32(-np.inf)), 448,
                        np.nextafter(np.float32(448), np.float32(np.inf))]
    boundaries[:, 1:5] = [1.0625, 1.1875, 1.3125, 1.4375]
    x = mx.array(np.concatenate([groups, boundaries]).reshape(1, 1, -1, 32)).astype(dtype)
    result = actual_heads(x)
    assert_bits(result, expected_for(x))
    assert_bits(_quantize_attention_output(x), result)


@pytest.mark.parametrize("reverse", [False, True])
def test_strided_sdpa_head_layout_and_cpu_fallback_match_gpu(reverse):
    values = np.random.default_rng(18).normal(size=(1, 64, 3, 512)).astype(np.float32)
    x = mx.array(values).astype(mx.bfloat16).transpose(0, 2, 1, 3)
    if reverse:
        x = x[..., ::-1]
    result = actual_heads(x)
    expected = expected_for(x)
    assert_bits(result, expected)
    mx.set_default_device(mx.cpu)
    try:
        # Exercise public CPU dispatch, not just the reference graph on GPU.
        assert_bits(actual_heads(x), expected)
    finally:
        mx.set_default_device(mx.gpu)


@pytest.mark.parametrize("length", [2, 5])
def test_quantized_heads_feed_native_grouped_projection_and_fused_epilogue(monkeypatch, length):
    rng = np.random.default_rng(9)
    x = mx.array(rng.normal(size=(1, length, 64, 512)).astype(np.float32)).astype(mx.bfloat16)
    grouped = actual_heads(x).reshape(1, length, 8, 4096)
    indices = np.arange(96) * 43 % 4096
    weight = np.zeros((8, 96, 4096), np.float32)
    weight[:, np.arange(96), indices] = 1
    weight = mx.array(weight).astype(mx.bfloat16)
    expected = expected_for(x).reshape(1, length, 8, 4096)[..., mx.array(indices)]
    monkeypatch.setattr(decode_fusions, "DS41_DECODE_KERNELS_V2", True)
    assert decode_fusions.grouped_gemv_supported(grouped, weight)
    assert_bits(decode_fusions.grouped_gemv(grouped, weight), expected)
    fused = attn_out.grouped_gemv_q(SimpleNamespace(weight=weight), grouped, weight, compact=False)
    assert_bits(fused, expected)
