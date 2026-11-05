# SPDX-License-Identifier: MIT
# Affine load/qdot arithmetic adapted from MLX (Copyright Apple Inc., MIT).
"""Native-engine LSQ3/G128 gate/up scheduling, with unchanged MLX QMV math.

This is a short decode/verify path for the current checkpoint, not a general
affine matmul. Gate and up occupy one dispatch, with route pairs advancing
fastest. Each output retains MLX's 16 values per lane, group scale/bias,
accumulation order, simd_sum, and BF16 output boundary. No weights are copied,
dequantized persistently, concatenated, or requantized.

The opt-in flag stays off until full-request tests establish a gain over the
current grouped-expert execution pipeline. The custom kernel remains lazy;
this module never evaluates arrays or reads route IDs back to the host.
"""

import os
from functools import cache

import mlx.core as mx

ENABLED = os.environ.get("DS41_NATIVE_GATE_UP", "0") == "1"
MAX_ROWS = 8
INPUT_DIM = 5120
OUTPUT_DIM = 2304
TOPK = 6

_HEADER = r"""
inline float ds41_native_load3(const device bfloat* x, thread float* v) {
    float sum = 0;
    for (int i = 0; i < 16; i += 8) {
        sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3] + x[i + 4]
             + x[i + 5] + x[i + 6] + x[i + 7];
        v[i] = x[i];
        v[i + 1] = x[i + 1] / 8.0f;
        v[i + 2] = x[i + 2] / 64.0f;
        v[i + 3] = x[i + 3] / 2.0f;
        v[i + 4] = x[i + 4] / 16.0f;
        v[i + 5] = x[i + 5] / 128.0f;
        v[i + 6] = x[i + 6] / 4.0f;
        v[i + 7] = x[i + 7] / 32.0f;
    }
    return sum;
}
inline float ds41_native_qdot3(const device uint8_t* w,
                              const thread float* v,
                              float scale, float bias, float sum) {
    float accum = 0;
    for (int i = 0; i < 2; i++) {
        v += 8 * i;
        w += 3 * i;
        accum += (w[0] & 0x07) * v[0];
        accum += (w[0] & 0x38) * v[1];
        accum += (w[0] & 0xc0) * v[2];
        accum += (w[1] & 0x01) * (v[2] * 256.0f);
        accum += (w[1] & 0x0e) * v[3];
        accum += (w[1] & 0x70) * v[4];
        accum += (w[1] & 0x80) * v[5];
        accum += (w[2] & 0x03) * (v[5] * 256.0f);
        accum += (w[2] & 0x1c) * v[6];
        accum += (w[2] & 0xe0) * v[7];
    }
    return scale * accum + sum * bias;
}
"""

_SOURCE = r"""
    const uint pair = threadgroup_position_in_grid.x;
    const uint block = threadgroup_position_in_grid.y;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const bool second = block >= N / 8;
    const uint row = (block % (N / 8)) * 8 + sg * 4;
    const uint expert = ids[pair];
    device bfloat* out = (second ? up : gate) + size_t(pair) * N + row;
    if (expert >= E) {
        if (lane == 0) for (int r = 0; r < 4; r++) out[r] = bfloat(0);
        return;
    }
    constexpr int weight_stride = K * 3 / 8;
    constexpr int scale_stride = K / 128;
    const device uint8_t* ws = (const device uint8_t*)(second ? w3 : w1)
        + (size_t(expert) * N + row) * weight_stride + lane * 6;
    const device bfloat* scales = (second ? s3 : s1)
        + (size_t(expert) * N + row) * scale_stride + lane / 8;
    const device bfloat* biases = (second ? b3 : b1)
        + (size_t(expert) * N + row) * scale_stride + lane / 8;
    const device bfloat* xs = x + size_t(pair / TOPK) * K + lane * 16;
    float result[4] = {0, 0, 0, 0};
    for (int k = 0; k < K; k += 512) {
        float v[16];
        const float sum = ds41_native_load3(xs, v);
        for (int r = 0; r < 4; r++) {
            const float scale = scales[r * scale_stride];
            const float bias = biases[r * scale_stride];
            result[r] += ds41_native_qdot3(
                ws + r * weight_stride, v, scale, bias, sum);
        }
        ws += 512 * 3 / 8;
        scales += 512 / 128;
        biases += 512 / 128;
        xs += 512;
    }
    for (int r = 0; r < 4; r++) {
        const float value = simd_sum(result[r]);
        if (lane == 0) out[r] = bfloat(value);
    }
"""


@cache
def _kernel():
    return mx.fast.metal_kernel(
        name="ds41_native_affine3_g128_gate_up",
        input_names=["x", "w1", "s1", "b1", "w3", "s3", "b3", "ids"],
        output_names=["gate", "up"],
        source=_SOURCE,
        header=_HEADER,
        ensure_row_contiguous=True,
    )


def _projection_ok(projection):
    if (getattr(projection, "mode", None) != "affine"
            or getattr(projection, "bits", None) != 3
            or getattr(projection, "group_size", None) != 128
            or not getattr(projection, "quantize_input", False)):
        return False
    w = projection.weight
    s = projection.scales
    b = projection.get("biases")
    return (
        w.ndim == 3 and 1 <= w.shape[0] <= 384 and w.dtype == mx.uint32
        and w.shape[1:] == (OUTPUT_DIM, INPUT_DIM * 3 // 32)
        and s.shape == (w.shape[0], OUTPUT_DIM, INPUT_DIM // 128)
        and s.dtype == mx.bfloat16 and b is not None
        and b.shape == s.shape and b.dtype == s.dtype
    )


def supported(expert, x, indices):
    """Only the exact BF16 short-routed geometry has been migrated."""
    if (not ENABLED or x.dtype != mx.bfloat16
            or mx.default_device() != mx.gpu or indices.ndim < 1
            or indices.dtype not in (mx.uint32, mx.int32)
            or indices.shape[-1] != TOPK or x.shape[-1] != INPUT_DIM):
        return False
    rows = indices.size // TOPK
    return (1 <= rows <= MAX_ROWS and x.size // INPUT_DIM == rows
            and _projection_ok(expert.w1) and _projection_ok(expert.w3)
            and expert.w1.weight.shape == expert.w3.weight.shape)


def gate_up(xq, w1, w3, ids, topk):
    """Project already quantized [rows,1,5120] inputs, without synchronizing.

    The caller owns activation quantization and the downstream SwiGLU/down
    path. Routes must be uint32 and token-major, six IDs per source token.
    Unsupported shapes fail before building any Metal kernel.
    """
    if (mx.default_device() != mx.gpu or xq.dtype != mx.bfloat16
            or xq.ndim != 3 or xq.shape[1:] != (1, INPUT_DIM)
            or not 1 <= xq.shape[0] <= MAX_ROWS or topk != TOPK
            or ids.ndim != 1 or ids.dtype != mx.uint32
            or ids.size != xq.shape[0] * TOPK
            or not _projection_ok(w1) or not _projection_ok(w3)
            or w1.weight.shape != w3.weight.shape):
        raise ValueError("Unsupported LSQ3/G128 paired gate/up geometry")
    outputs = (ids.size, 1, OUTPUT_DIM)
    return _kernel()(
        inputs=[xq, w1.weight, w1.scales, w1.biases,
                w3.weight, w3.scales, w3.biases, ids],
        template=[("K", INPUT_DIM), ("N", OUTPUT_DIM),
                  ("E", w1.weight.shape[0]), ("TOPK", TOPK)],
        grid=(32 * ids.size, 2 * (OUTPUT_DIM // 8) * 2, 1),
        threadgroup=(32, 2, 1),
        output_shapes=[outputs, outputs],
        output_dtypes=[mx.bfloat16, mx.bfloat16],
    )
