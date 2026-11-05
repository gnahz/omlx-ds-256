# SPDX-License-Identifier: MIT
"""Opt-in expert-sorted NAX prefill with dictionary metadata in weight tiles.

The native primitive retains MLX's BF16 tile dequantization and FP32 NAX
accumulation. Only scale/bias addressing changes. Each 64-row expert tile reuses
the decoded weight tile without a dense metadata bank or a local eval barrier.
Short/tail blocks retain the qualified native_metadata fallbacks.
"""
import os

import mlx.core as mx

from omlx.custom_kernels.glm_moe_dsa import fast
from .activation import quantize_swiglu_activation
from .quantization import quantize_activation
from .routing import combine_sorted_experts

FLAG = "DS41_NATIVE_METADATA_PREFILL"
ENABLED = os.environ.get(FLAG, "0") == "1"
SYMBOL = "deepseek_v41_joint_metadata_gather_qmm"


def eligible(moe, x):
    # Admission stays on the existing >=512 source-row gather path. Below that
    # boundary the original dispatch may use different QMV/QMM arithmetic.
    from .native_metadata import JointMetadataProjection

    if (not ENABLED or not fast.has_symbol(SYMBOL)
            or mx.default_device() != mx.gpu or x.dtype != mx.bfloat16
            or x.ndim != 3 or x.shape[0] != 1 or x.shape[1] < 512):
        return False
    projections = (moe.experts.w1, moe.experts.w3, moe.experts.w2)
    if not all(isinstance(p, JointMetadataProjection) and p.quantize_input
               and p.input_dims % 128 == 0 and p.output_dims % 64 == 0
               and p.input_dims <= 65536 and p.output_dims <= 65536
               for p in projections):
        return False
    gate, up, down = projections
    experts = gate.weight.shape[0]
    topk = getattr(getattr(moe.gate, "_config", None), "n_activated_experts", None)
    return (topk == 6 and 1 <= experts <= 384
            and x.shape[1] * topk // experts >= 4
            and gate.weight.shape == up.weight.shape
            and down.weight.shape[0] == experts
            and gate.input_dims == down.output_dims == x.shape[-1]
            and gate.output_dims == down.input_dims)


def forward(moe, x, image_mask):
    """Preserve the original router, source rows, FP8 activations and combine."""
    ids, weights = moe.gate(x, image_mask)
    if ids.shape != (1, x.shape[1], 6) or weights.shape != ids.shape:
        raise ValueError("Unexpected native metadata prefill routing geometry")
    shape = ids.shape
    flat = ids.reshape(-1)
    order = mx.argsort(flat)
    inverse = mx.argsort(order)
    expert = flat[order].astype(mx.uint32)
    rows = (order // shape[-1]).astype(mx.uint32)
    source = quantize_activation(x).reshape(-1, 1, x.shape[-1])

    def gather(projection, value, lhs=None):
        return fast.deepseek_v41_joint_metadata_gather_qmm(
            value, projection.weight, projection.joint_words,
            projection.joint_dictionary, lhs, expert)

    gate = gather(moe.experts.w1, source, rows)
    up = gather(moe.experts.w3, source, rows)
    hidden = quantize_swiglu_activation(
        gate, up, weights.reshape(-1)[order], x.dtype, moe.experts._limit)
    routed = gather(moe.experts.w2, hidden)
    shared = moe.shared_experts(x)
    combined = combine_sorted_experts(routed, inverse, shared)
    if combined is not None:
        return combined
    routed = routed[inverse].reshape(*shape, x.shape[-1])
    return (routed.astype(mx.float32).sum(-2)
            + shared.astype(mx.float32)).astype(x.dtype)
