"""Qualified DeepSeek names backed by the current split native libraries."""
from ..attention import fast as attention
from ..quant_matmul import fast as quant
from ..sparse_index import fast as sparse

_ALIASES = {
    'dsa_decode_scores': (sparse, 'dsa_decode_scores'),
    'dsa_indexer_scores': (sparse, 'dsa_indexer_scores'),
    'dsa_indexer_scores_mma': (sparse, 'dsa_indexer_scores_mma'),
    'qwen4_qsa_indexer_scores': (sparse, 'qsa_indexer_scores'),
    'qwen4_qsa_topk_indices': (sparse, 'qsa_topk_indices'),
    'dsa_topk_indices': (sparse, 'dsa_topk_indices'),
    'dspark_fp32_topk_indices': (sparse, 'fp32_topk_indices'),
    'dspark_exact_mxfp8_qmv_pair': (quant, 'mxfp8_qmv_pair'),
    'glm_dsa_sparse_mla_attention': (attention, 'sparse_mla_attention'),
    'glm_dsa_exact_block_attention': (attention, 'block_mask_attention'),
    'deepseek_v4_sparse_attention': (attention, 'compressed_sparse_attention'),
    'deepseek_v41_packed_attention': (attention, 'packed_sparse_attention'),
    'deepseek_v41_grouped_expert': (quant, 'grouped_expert'),
    'deepseek_v41_joint_metadata_gather_qmm': (quant, 'deepseek_v41_joint_metadata_gather_qmm'),
    'dspark_ring_gemm': (attention, 'ring_gemm'),
    'dspark_rowwise_gemm': (attention, 'rowwise_gemm'),
    'glm_dsa_q8_vup_flat': (quant, 'q8_head_flat_qmm'),
    'glm_moe_weighted_sum': (quant, 'moe_weighted_sum'),
    'qwen4_qsa_sparse_gqa_attention': (attention, 'sparse_gqa_attention'),
    'deepseek_mxfp4_gather_qmm_blocks': (quant, 'mxfp4_gather_qmm_blocks'),
    'deepseek_mxfp4_gather_qmm_pair_blocks': (quant, 'mxfp4_gather_qmm_pair_blocks'),
    'deepseek_mxfp4_gather_qmm_pair_concat_blocks': (quant, 'mxfp4_gather_qmm_pair_concat_blocks'),
    'deepseek_mxfp4_gather_qmm_expert': (quant, 'mxfp4_gather_qmm_expert'),
    'deepseek_affine_gather_qmm_blocks': (quant, 'affine_gather_qmm_blocks'),
    'deepseek_affine_gather_qmm_pair_concat_blocks': (quant, 'affine_gather_qmm_pair_concat_blocks'),
}
NATIVE_SYMBOLS = tuple(_ALIASES)


def has_symbol(name):
    module, target = _ALIASES.get(name, (quant, name))
    return module.has_symbol(target)


def native_symbols():
    return tuple(name for name in NATIVE_SYMBOLS if has_symbol(name))


def missing_symbols(required):
    return [name for name in required if not has_symbol(name)]


def is_native_available():
    return all(module.is_native_available() for module in (attention, quant, sparse))


def import_error():
    return next((module.import_error() for module in (attention, quant, sparse)
                 if module.import_error() is not None), None)


def __getattr__(name):
    if name == "_IMPORT_ERROR":
        return import_error()
    if name in _ALIASES:
        module, target = _ALIASES[name]
        return getattr(module, target)
    for module in (quant, attention, sparse):
        if hasattr(module, name):
            return getattr(module, name)
    raise AttributeError(name)


class _ExtensionProxy:
    def __getattr__(self, name):
        module, target = _ALIASES.get(name, (quant, name))
        return getattr(module._ext, target)


_ext = _ExtensionProxy() if is_native_available() else None
