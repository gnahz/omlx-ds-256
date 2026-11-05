"""Actual dispatch/planner/profile AST with no MLX import or model allocation."""
from __future__ import annotations

import ast
from collections.abc import Sequence
from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
from types import SimpleNamespace as NS
from typing import Any

import pytest

SOURCE = Path(__file__).resolve().parents[1]
MODEL_CONFIG = SOURCE / "tests/fixtures/deepseek_v41_1017_config.json"
MIB = 1024**2


def extract(path, names, namespace):
    nodes = [node for node in ast.parse(path.read_text()).body
             if getattr(node, "name", None) in names]
    assert {node.name for node in nodes} == set(names)
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[])),
                 str(path), "exec"), namespace)


@pytest.fixture(scope="module")
def cpp_plan(tmp_path_factory):
    path = tmp_path_factory.mktemp("workspace-plan")
    header = SOURCE / "omlx/custom_kernels/attention/csrc/packed_sparse_attention_plan.h"
    main = path / "plan.cpp"
    main.write_text(
        '#include <iostream>\n#include "' + str(header) + '"\n'
        'int main(int argc,char**argv){try{auto p=omlx::attention::packed_sparse_attention_workspace('
        'std::stoll(argv[1]),std::stoll(argv[2]),std::stoll(argv[3]),std::stoll(argv[4]));'
        'std::cout<<p.unpack_kv<<" "<<p.unique_kv_bytes<<" "<<p.scratch_bytes;return 0;}'
        'catch(const std::exception&e){std::cerr<<e.what();return 2;}}\n')
    binary = path / "plan"
    subprocess.run(["/usr/bin/c++", "-std=c++17", "-O2", str(main), "-o", str(binary)], check=True)

    def query(batch, queries, local_rows, pooled_rows):
        values = subprocess.check_output(
            [str(binary), *map(str, (batch, queries, local_rows, pooled_rows))], text=True)
        unpack, unique, scratch = map(int, values.split())
        return {"abi": 1, "unpack_kv": bool(unpack), "unique_kv_bytes": unique,
                "scratch_bytes": scratch, "scratch_limit_bytes": 80 * MIB}
    query.binary = binary
    return query


@pytest.fixture
def policy(cpp_plan, monkeypatch):
    ext = NS(PACKED_SPARSE_WORKSPACE_ABI=1, packed_sparse_attention=object(),
             packed_sparse_attention_single_decode=object(),
             packed_sparse_attention_workspace=cpp_plan)
    fast = {"_ext": ext, "has_symbol": lambda name: hasattr(ext, name)}
    extract(SOURCE / "omlx/custom_kernels/attention/fast.py",
            {"deepseek_v41_packed_attention_route", "deepseek_v41_prefill_workspace_descriptor"}, fast)
    kernels = {"DS41_PREFILL_MID_INDEX": True, "DS41_DECODE_SINGLE_TILE": True,
               "_RADIX_MIN_KEYS": 4096}
    workspace = runpy.run_path(str(SOURCE / "omlx/patches/deepseek_v41/workspace.py"))
    kernels["_workspace"] = NS(**workspace)
    extract(SOURCE / "omlx/patches/deepseek_v41/kernels.py",
            {"index_radix_query_group", "index_scan_plan", "packed_index_workspace_bytes"}, kernels)
    config_ns = runpy.run_path(str(SOURCE / "omlx/patches/deepseek_v41/config.py"))
    raw = json.loads(MODEL_CONFIG.read_text())
    config = replace(config_ns["ModelConfig"].from_dict(raw), ced_prefill=True)
    config._native_metadata_prefill_transient_bytes = 424_673_280
    monitor = {"dataclass": dataclass, "Any": Any, "Sequence": Sequence,
               "_v41_index_workspace_bytes": workspace["packed_index_workspace_bytes"]}
    extract(SOURCE / "omlx/memory_monitor.py", {
        "_cfg_get", "_pos_int", "_DeepSeekV41PrefillMemoryProfile",
        "_make_deepseek_v41_prefill_memory_profile"}, monitor)
    language_path = SOURCE / "omlx/patches/deepseek_v41/language.py"
    language_class = next(node for node in ast.parse(language_path.read_text()).body
                          if isinstance(node, ast.ClassDef) and node.name == "LanguageModel")
    method = next(node for node in language_class.body
                  if getattr(node, "name", None) == "_omlx_native_prefill_workspace_descriptor")
    class PlainEmbedding:
        def __init__(self):
            self.weight = NS(dtype="bf16")

    class PackedEmbedding:
        mode, quantize_input = "affine", False
        weight = NS(dtype="uint32")
        scales, biases = NS(dtype="bf16"), NS(dtype="bf16")

    language = {"os": os, "attention_fast": NS(**fast),
                "mx": NS(gpu="gpu", bfloat16="bf16", default_device=lambda: "gpu"),
                "nn": NS(Embedding=PlainEmbedding), "QuantizedEmbedding": PackedEmbedding,
                "packed_index_workspace_bytes": kernels["packed_index_workspace_bytes"]}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(language_path), "exec"), language)
    host = NS(_config=config, pipeline_size=1,
              layers=[NS(_tensor_group=None) for _ in range(config.n_layers)],
              embed=PlainEmbedding())
    monkeypatch.setenv("DS41_NATIVE_PREFILL_WORKSPACE", "1")
    descriptor = language["_omlx_native_prefill_workspace_descriptor"](host)
    assert descriptor is not None
    factory = monitor["_make_deepseek_v41_prefill_memory_profile"]
    return NS(ext=ext, fast=fast, kernels=kernels, config=config, host=host,
              language=language, descriptor=descriptor, factory=factory,
              packed_embedding=PackedEmbedding,
              native=factory(config, compute_dtype_size=2, deepseek_native_attention=descriptor),
              fallback=factory(config, compute_dtype_size=2))


@pytest.mark.parametrize("queries,local,pooled", [
    (128, 128, 262144), (256, 256, 0), (8192, 8320, 32768),
    (8192, 8192, 73728), (8192, 8192, 73729), (8192, 8320, 524288),
])
def test_cpp_workspace_plan_matches_actual_unique_kv_and_eighty_mib_switch(cpp_plan, queries, local, pooled):
    result = cpp_plan(1, queries, local, pooled)
    unique = (local + max(pooled, 1)) * 512 * 2
    assert result["unique_kv_bytes"] == unique
    assert result["unpack_kv"] == (queries >= 256 and unique <= 80 * MIB)
    assert result["scratch_bytes"] == (unique if result["unpack_kv"] else 0)
    assert result["scratch_bytes"] <= 80 * MIB


@pytest.mark.parametrize("values", [(0, 8192, 8192, 0), (1, 0, 8192, 0),
                                  (1, 8192, 100, 0), (1, 8192, 8192, -1)])
def test_cpp_workspace_invalid_geometry_fails_closed(cpp_plan, values):
    result = subprocess.run([str(cpp_plan.binary), *map(str, values)], capture_output=True)
    assert result.returncode == 2


def test_workspace_query_does_not_exist_in_old_extension_so_pricing_stays_fallback(policy):
    del policy.ext.packed_sparse_attention_workspace
    assert policy.language["_omlx_native_prefill_workspace_descriptor"](policy.host) is None
    assert policy.fallback.prediction_components(8192, 65536)["attention_gather_bytes"] == 5 * 1024**3


@pytest.mark.parametrize("change", [
    {"batch_size": 2}, {"num_heads": 32}, {"head_dim": 256},
    {"is_gpu": False}, {"is_bfloat16": False},
])
def test_native_discount_requires_exact_runtime_geometry_and_device_dtype(policy, change):
    args = dict(batch_size=1, num_heads=64, head_dim=512, is_gpu=True, is_bfloat16=True)
    args.update(change)
    assert policy.fast["deepseek_v41_prefill_workspace_descriptor"](**args) is None


def test_workspace_query_is_opt_in_until_peak_qualification(policy, monkeypatch):
    monkeypatch.delenv("DS41_NATIVE_PREFILL_WORKSPACE")
    assert policy.language["_omlx_native_prefill_workspace_descriptor"](policy.host) is None


@pytest.mark.parametrize("change", [
    {"local_row_bytes": 512}, {"pooled_row_bytes": 256},
    {"local_rows": 4096}, {"query_tokens": 8}, {"pooled_rows": -1},
])
def test_shared_native_route_rejects_wrong_packing_and_unsupported_shapes(policy, change):
    args = dict(query_tokens=8192, batch_size=1, num_heads=64, head_dim=512,
                is_gpu=True, is_bfloat16=True, local_rows=8192, pooled_rows=32768)
    args.update(change)
    assert policy.fast["deepseek_v41_packed_attention_route"](**args) is None


@pytest.mark.parametrize("tokens,context", [(1024, 1024), (4096, 65536),
                                          (8192, 65536), (8192, 263215), (8192, 1048576)])
def test_native_pricing_replaces_only_gather_and_retains_other_charges(policy, tokens, context):
    native = policy.native.prediction_components(tokens, context)
    fallback = policy.fallback.prediction_components(tokens, context)
    assert native["native_attention_workspace"] is True
    assert native["attention_gather_bytes"] == 0
    assert native["native_attention_scratch_bytes"] <= 80 * MIB
    assert native["logical_kv_growth_bytes"] == fallback["logical_kv_growth_bytes"] == 890 * tokens
    for field in ("query_output_bytes", "index_projection_bytes", "hyperconnection_bytes",
                  "pipeline_overlap_bytes",
                  "moe_bytes", "metadata_fallback_bytes"):
        assert native[field] == fallback[field]
    assert native["metadata_fallback_bytes"] == 424_673_280
    assert native["index_scratch_bytes"] > 0
    assert fallback["index_scratch_bytes"] > 0
    assert native["pipeline_overlap_bytes"] > 0
    charged = [value for key, value in native.items()
               if key.endswith("_bytes") and key not in ("transient_bytes", "logical_kv_growth_bytes")]
    assert native["transient_bytes"] == sum(charged)
    assert native["transient_bytes"] < fallback["transient_bytes"]


def test_midpoint_two_group_overlap_and_selection_are_preserved(policy):
    charge = policy.kernels["packed_index_workspace_bytes"](
        128, 1048576, 512, block_count=2048, block_size=8)
    # Score tiles alone can occupy two 128 MiB groups, before their winners.
    assert charge >= 256 * MIB
    parts = policy.native.prediction_components(8192, 1048576)
    assert parts["index_scratch_bytes"] >= charge


@pytest.mark.parametrize("queries", [1, 8])
def test_short_queries_keep_tile_winners_and_bound_both_decode_flag_states(policy, queries):
    workspace = policy.kernels["_workspace"]
    context, count = 1048576, 512
    bound = workspace.packed_index_workspace_bytes(queries, context, count)
    for mid_index in (False, True):
        for single_tile in (False, True):
            rows, width = workspace.index_scan_plan(
                queries, context, mid_index=mid_index, decode_single_tile=single_tile)
            rows, width = min(rows, queries), min(width, context)
            limit = max(1024, 2 * count)
            tile = max(32, 1 << (min(width, limit) - 1).bit_length())
            tiles = (width + tile - 1) // tile
            required = 2 * rows * width * 4 + 3 * rows * tiles * count * 8
            assert bound >= required
    # A radix-only score allowance omits these substantial tile winners.
    assert bound > 2 * queries * context * 4


def test_pinned_image_embedding_path_preserves_bf16_query_dtype_proof():
    model = (SOURCE / "omlx/patches/deepseek_v41/model.py").read_text()
    assert "h = self.language_model.embed(input_ids)" in model
    assert "mx.stack(values).astype(h.dtype)" in model


def test_quantized_embedding_output_dtype_not_u32_storage_proves_native_route(policy):
    policy.host.embed = policy.packed_embedding()
    assert policy.host.embed.weight.dtype == "uint32"
    descriptor = policy.language["_omlx_native_prefill_workspace_descriptor"](policy.host)
    assert descriptor is not None
    policy.host.embed.scales = policy.host.embed.biases = NS(dtype="float32")
    assert policy.language["_omlx_native_prefill_workspace_descriptor"](policy.host) is None


def test_unknown_embedding_implementation_retains_conservative_fallback(policy):
    policy.host.embed = NS(weight=NS(dtype="bf16"))
    assert policy.language["_omlx_native_prefill_workspace_descriptor"](policy.host) is None


def test_bad_workspace_provider_never_reduces_estimate(policy):
    descriptor = {**policy.descriptor, "workspace": lambda *args: {"abi": 1, "attention_scratch_bytes": -1}}
    profile = policy.factory(policy.config, compute_dtype_size=2, deepseek_native_attention=descriptor)
    assert profile.estimate_prefill_transient_bytes(8192, 65536) == (
        policy.fallback.estimate_prefill_transient_bytes(8192, 65536))


def test_cpp_allocation_and_python_dispatch_use_the_same_planners():
    cpp = (SOURCE / "omlx/custom_kernels/attention/csrc/packed_sparse_attention.cpp").read_text()
    language = (SOURCE / "omlx/patches/deepseek_v41/language.py").read_text()
    kernels = (SOURCE / "omlx/patches/deepseek_v41/kernels.py").read_text()
    assert "packed_sparse_attention_workspace(B, qL, localL, pooledL)" in cpp
    assert "attention_fast.deepseek_v41_packed_attention_route(" in language
    assert "group = index_radix_query_group(keys.shape[1])" in kernels
    assert "query_chunk_size, chunk_size = index_scan_plan(" in kernels


def test_cpu_policy_tests_import_no_compute_runtime():
    assert not any(name == "mlx" or name.startswith("mlx.") for name in sys.modules)
