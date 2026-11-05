"""Attention kernels: sparse, block-mask, FA-256, decode SDPA and verify GEMM."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlx.core as mx

from .._loader import load_extension
from .._loader import native_stream_kwargs as _native_stream_kwargs

_ext, _IMPORT_ERROR = load_extension(__package__, Path(__file__).parent)


def is_native_available() -> bool:
    return _ext is not None


def import_error() -> Exception | None:
    return _IMPORT_ERROR


def has_symbol(name: str) -> bool:
    return hasattr(_ext, name)


NATIVE_AVAILABLE = _ext is not None


def deepseek_v41_packed_attention_route(
    *, query_tokens, batch_size, num_heads, head_dim, is_gpu, is_bfloat16,
    local_rows, pooled_rows, local_row_bytes=528, pooled_row_bytes=288,
    selected_rows=0, native_decode=True, native_verify=False,
):
    """One metadata-only route predicate shared by dispatch and admission."""
    geometry = (query_tokens, batch_size, num_heads, head_dim, local_rows, pooled_rows)
    if (any(type(value) is not int for value in geometry)
            or query_tokens <= 0 or batch_size <= 0
            or num_heads != 64 or head_dim != 512 or local_rows < query_tokens
            or pooled_rows < 0 or local_row_bytes != 528 or pooled_row_bytes != 288
            or is_gpu is not True or is_bfloat16 is not True):
        return None
    if (native_decode and query_tokens == 1 and selected_rows == 0
            and pooled_rows == 0 and has_symbol("packed_sparse_attention_single_decode")):
        return "single"
    if ((query_tokens > 8 or (native_verify and query_tokens > 1))
            and has_symbol("packed_sparse_attention")):
        return "multi"
    return None


def deepseek_v41_prefill_workspace_descriptor(
    *, batch_size, num_heads, head_dim, is_gpu, is_bfloat16,
):
    """Absent/stale native extensions retain conservative fallback pricing."""
    if (batch_size != 1 or getattr(_ext, "PACKED_SPARSE_WORKSPACE_ABI", None) != 1
            or not has_symbol("packed_sparse_attention_workspace")
            or deepseek_v41_packed_attention_route(
                query_tokens=256, batch_size=batch_size, num_heads=num_heads,
                head_dim=head_dim, is_gpu=is_gpu, is_bfloat16=is_bfloat16,
                local_rows=256, pooled_rows=0) != "multi"):
        return None

    def workspace(query_tokens, local_rows, pooled_rows):
        if deepseek_v41_packed_attention_route(
                query_tokens=query_tokens, batch_size=1, num_heads=num_heads,
                head_dim=head_dim, is_gpu=is_gpu, is_bfloat16=is_bfloat16,
                local_rows=local_rows, pooled_rows=pooled_rows) != "multi":
            return None
        result = _ext.packed_sparse_attention_workspace(
            1, query_tokens, local_rows, pooled_rows)
        if (not isinstance(result, dict) or result.get("abi") != 1
                or type(result.get("scratch_bytes")) is not int
                or not 0 <= result["scratch_bytes"] <= 80 * 1024**2):
            return None
        return result

    return {"abi": 1, "workspace": workspace, "batch_size": 1,
            "num_heads": num_heads, "head_dim": head_dim}


def sparse_mla_attention(
    q_latent: mx.array,
    q_pe: mx.array,
    kv_latent: mx.array,
    k_pe: mx.array,
    topk_indices: mx.array,
    scale: float,
    causal: bool = True,
    topk_valid_prefix: bool = False,
    causal_prefix_indices: bool = False,
    topk_length: mx.array | None = None,
    causal_prefix_rows: int = 0,
    *,
    stream=None,
) -> mx.array:
    if _ext is not None:
        return _ext.sparse_mla_attention(
            q_latent,
            q_pe,
            kv_latent,
            k_pe,
            topk_indices,
            scale,
            causal=causal,
            topk_valid_prefix=topk_valid_prefix,
            causal_prefix_indices=causal_prefix_indices,
            topk_length=topk_length,
            causal_prefix_rows=causal_prefix_rows,
            **_native_stream_kwargs(stream),
        )
    raise RuntimeError("sparse_mla_attention native kernel is unavailable")


def block_mask_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    block_mask: mx.array,
    block_token_mask: mx.array,
    scale: float,
    causal: bool = True,
    *,
    stream=None,
) -> mx.array:
    if _ext is not None and hasattr(_ext, "block_mask_attention"):
        return _ext.block_mask_attention(
            q,
            k,
            v,
            block_mask,
            block_token_mask,
            scale,
            causal=causal,
            **_native_stream_kwargs(stream),
        )
    raise RuntimeError("block_mask_attention native kernel is unavailable")


def compressed_sparse_attention(
    q: mx.array,
    local_kv: mx.array,
    pooled: mx.array,
    topk_indices: mx.array,
    sinks: mx.array,
    scale: float,
    q_offset: int,
    compress_ratio: int,
    local_window: int,
    *,
    stream=None,
) -> mx.array:
    if _ext is not None and hasattr(_ext, "compressed_sparse_attention"):
        return _ext.compressed_sparse_attention(
            q,
            local_kv,
            pooled,
            topk_indices,
            sinks,
            scale,
            q_offset,
            compress_ratio,
            local_window,
            **_native_stream_kwargs(stream),
        )
    raise RuntimeError("compressed_sparse_attention native kernel is unavailable")


def packed_sparse_attention(
    q: mx.array,
    local_kv: mx.array,
    pooled: mx.array,
    topk_indices: mx.array,
    sinks: mx.array,
    scale: float,
    q_offset: int,
    compress_ratio: int,
    local_window: int,
    *,
    stream=None,
) -> mx.array:
    if _ext is not None and hasattr(_ext, "packed_sparse_attention"):
        return _ext.packed_sparse_attention(
            q,
            local_kv,
            pooled,
            topk_indices,
            sinks,
            scale,
            q_offset,
            compress_ratio,
            local_window,
            **_native_stream_kwargs(stream),
        )
    raise RuntimeError("packed_sparse_attention native kernel is unavailable")

def packed_sparse_attention_single_decode(
    q: mx.array,
    local_kv: mx.array,
    pooled: mx.array,
    topk_indices: mx.array,
    sinks: mx.array,
    scale: float,
    q_offset: int,
    compress_ratio: int,
    local_window: int,
    *,
    stream=None,
) -> mx.array:
    if _ext is not None and hasattr(_ext, "packed_sparse_attention_single_decode"):
        return _ext.packed_sparse_attention_single_decode(
            q,
            local_kv,
            pooled,
            topk_indices,
            sinks,
            scale,
            q_offset,
            compress_ratio,
            local_window,
            **_native_stream_kwargs(stream),
        )
    raise RuntimeError("packed_sparse_attention_single_decode native kernel is unavailable")



def sparse_gqa_attention(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    selected_blocks: mx.array,
    scale: float,
    q_offset: int,
    *,
    key_tile: int = 128,
    dimension_tile: int = 32,
    stream=None,
) -> mx.array:
    """Exact Qwen4 main GQA over query-specific selected cache rows.

    The narrow native ABI consumes 512 chronological uint32 block IDs directly,
    expands each to four tokens, appends the causal tail in-kernel, and
    keeps QK scores, online softmax state, and the weighted output in fp32.
    It supports only Qwen3.8-Flash-Next's batch-one ``24q/2kv/D256`` geometry;
    callers own the portable gathered fallback for every other shape.
    """

    if _ext is None or not hasattr(_ext, "sparse_gqa_attention"):
        raise RuntimeError("sparse_gqa_attention native kernel is unavailable")
    return _ext.sparse_gqa_attention(
        queries,
        keys,
        values,
        selected_blocks,
        scale,
        q_offset,
        key_tile,
        dimension_tile,
        **_native_stream_kwargs(stream),
    )


def roll_rows(
    x: mx.array,
    shifts: mx.array,
    axis: int,
    length: int,
    max_shift: int,
    *,
    stream=None,
) -> mx.array:
    if _ext is None or not hasattr(_ext, "roll_rows"):
        raise RuntimeError("roll_rows native kernel is unavailable")
    return _ext.roll_rows(
        x,
        shifts,
        axis,
        length,
        max_shift,
        **_native_stream_kwargs(stream),
    )


def rowwise_gemm(
    lhs: mx.array,
    rhs: mx.array,
    transpose_rhs: bool,
    *,
    stream=None,
) -> mx.array:
    if _ext is None or not hasattr(_ext, "rowwise_gemm"):
        raise RuntimeError("rowwise_gemm native kernel is unavailable")
    return _ext.rowwise_gemm(
        lhs,
        rhs,
        transpose_rhs,
        **_native_stream_kwargs(stream),
    )


def ring_gemm(
    lhs: mx.array,
    source: mx.array,
    indices: mx.array,
    transpose_rhs: bool,
    *,
    stream=None,
) -> mx.array:
    if _ext is None or not hasattr(_ext, "ring_gemm"):
        raise RuntimeError("ring_gemm native kernel is unavailable")
    return _ext.ring_gemm(
        lhs,
        source,
        indices,
        transpose_rhs,
        **_native_stream_kwargs(stream),
    )


# Extensions built before the chunked-dispatch fix (issue #2225) reject the
# dispatch_budget kwarg, so only forward it when the rebuilt binding is
# present; those older builds keep the single-dispatch behavior.
_EXT_HAS_FA256_DISPATCH_BUDGET = _ext is not None and hasattr(
    _ext, "FA256_HAS_DISPATCH_BUDGET"
)


def fa256_supports_dispatch_budget() -> bool:
    return _EXT_HAS_FA256_DISPATCH_BUDGET


def fa256_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    scale: float,
    causal: bool = True,
    q_block: int = 32,
    k_block: int = 8,
    dispatch_budget: int = 0,
    *,
    stream=None,
) -> mx.array:
    if _ext is not None and hasattr(_ext, "fa256_attention"):
        budget_kwargs: dict[str, int] = {}
        if _EXT_HAS_FA256_DISPATCH_BUDGET:
            budget_kwargs["dispatch_budget"] = int(dispatch_budget)
        return _ext.fa256_attention(
            q,
            k,
            v,
            scale,
            causal=causal,
            q_block=q_block,
            k_block=k_block,
            **budget_kwargs,
            **_native_stream_kwargs(stream),
        )
    raise RuntimeError("fa256_attention native kernel is unavailable")


def sdpa_decode(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    scale: float,
    causal: bool = False,
    mask: mx.array | None = None,
    sinks: mx.array | None = None,
    *,
    stream: mx.Stream | None = None,
    force_fallback: bool = False,
) -> mx.array:
    """Run native decode SDPA when its exact ABI accepts the inputs."""
    if (
        not force_fallback
        and _ext is not None
        and _ext.sdpa_decode_supported(q, k, v, stream)
    ):
        return _ext.sdpa_decode(q, k, v, scale, causal, mask, sinks, stream)
    return mx.fast.scaled_dot_product_attention(
        q,
        k,
        v,
        scale=scale,
        mask="causal" if causal and mask is None else mask,
        sinks=sinks,
    )


def __getattr__(name: str) -> Any:
    if _ext is not None and hasattr(_ext, name):
        return getattr(_ext, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    names = set(globals())
    if _ext is not None:
        names.update(dir(_ext))
    return sorted(names)
