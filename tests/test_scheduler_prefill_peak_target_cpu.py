"""Extract scheduler decisions without importing MLX or running any model."""
from __future__ import annotations

import ast
from dataclasses import dataclass
import math
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import pytest

GIB = 1024**3
SOURCE = Path(__file__).resolve().parents[1] / "omlx/scheduler.py"


@pytest.fixture(scope="module")
def definitions():
    tree = ast.parse(SOURCE.read_text())
    config = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "SchedulerConfig")
    scheduler = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Scheduler")
    names = {"_prefill_abort_cap", "_prefill_peak_target_caps", "_prefill_peak_target",
             "_adaptive_chunk_size", "_largest_fitting_prefill_chunk", "_snap_chunk_size",
             "_deepseek_native_attention_descriptor", "_record_prefill_chunk_decision",
             "_record_prefill_cache_storage", "_guard_prefill_chunk", "_prefill_abort_description",
             "_admission_transient_bound", "_admission_limit_bytes",
             "_maybe_upgrade_prefill_budget"}
    nodes = [node for node in scheduler.body if getattr(node, "name", None) in names]
    assert {node.name for node in nodes} == names
    subset = ast.ClassDef(name="Scheduler", bases=[], keywords=[], body=nodes, decorator_list=[])
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    namespace = {"__name__": __name__, "dataclass": dataclass, "math": math,
                 "SchedulingPolicy": NS(FCFS="fcfs"),
                 "_exact_prefix_manager": lambda model: None,
                 "describe_ceiling_binding": lambda **kwargs: ("live", "close other apps"),
                 "format_bytes": str, "logger": NS(info=lambda *args, **kwargs: None,
                                                   debug=lambda *args, **kwargs: None)}
    before = set(sys.modules)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, config, subset], type_ignores=[])),
                 str(SOURCE), "exec"), namespace)
    assert not any(name == "mlx" or name.startswith("mlx.") for name in set(sys.modules) - before)
    cls = namespace["Scheduler"]
    cls._PREFILL_HEADROOM_SAFETY = .90
    cls._PREFILL_STEP_TIERS = (1024, 512)
    cls._PREFILL_TRANSIENT_SAFETY = 1.3
    return NS(config=namespace["SchedulerConfig"], scheduler=cls, tree=tree)


def decision(definitions, *, requested=236, hard=240, abort=None, current=220, prediction=10.78):
    sched = definitions.scheduler()
    sched.config = definitions.config(prefill_peak_target_gib=requested)
    sched._memory_hard_limit_bytes = int(hard * GIB)
    sched._memory_abort_limit_bytes = int((hard if abort is None else abort) * GIB)
    sched._prefill_headroom_safety = sched._prefill_abort_margin = .995
    sched._memory_limit_bytes = int(hard * .97 * GIB)
    sched._memory_hard_watermark_bytes = int(hard * .999 * GIB)
    sched._prefill_safe_zone_ratio = .99
    sched._prefill_min_chunk_tokens = 32
    sched._prefill_speed_priority = False
    sched._current_usage_bytes = lambda: int(current * GIB)
    sched._reclaim_prefill_headroom = sched._current_usage_bytes
    sched._predicted_chunk_transient = lambda n, *args, **kwargs: n * prediction * GIB / 8192
    sched._raise_prefill_eviction_if_available = lambda **kwargs: None
    sched.model = None
    sched.memory_monitor = None
    sched._prefill_transient_tracker = None
    sched._throttle_notified_requests = set()
    sched._memory_static_ceiling_bytes = 254 * GIB
    sched._memory_dynamic_ceiling_bytes = int(hard * GIB)
    sched._memory_metal_cap_bytes = 240 * GIB
    sched._memory_metal_cap_raw_bytes = 240 * GIB
    sched._memory_guard_tier = "custom"
    return sched


@pytest.mark.parametrize("value", [0, -1, True, "236", float("nan"), float("inf")])
def test_config_rejects_invalid_target(definitions, value):
    with pytest.raises(ValueError, match="prefill_peak_target_gib"):
        definitions.config(prefill_peak_target_gib=value)


def test_default_none_preserves_previous_target(definitions):
    sched = decision(definitions, requested=None)
    assert sched._prefill_peak_target() == int(240 * GIB * .995)
    assert sched._last_prefill_peak_caps["requested_target_bytes"] == 0


def test_236_upper_target_preserves_live_physical_and_abort_caps(definitions):
    sched = decision(definitions)
    assert sched._prefill_peak_target() == 236 * GIB
    assert sched._last_prefill_peak_caps["binding_limits"] == ["requested"]
    low = decision(definitions, hard=226.593)
    assert low._prefill_peak_target() == int(int(226.593 * GIB) * .995)
    assert low._prefill_peak_target() < 236 * GIB
    assert "headroom" in low._last_prefill_peak_caps["binding_limits"]
    abort = decision(definitions, hard=240, abort=228)
    assert abort._prefill_peak_target() == int(228 * GIB * .995)
    assert abort._last_prefill_peak_caps["binding_limits"] == ["abort"]


def test_upper_target_cannot_replace_uninitialised_live_guard(definitions):
    sched = decision(definitions, hard=0)
    assert sched._prefill_peak_target() == 0
    assert sched._last_prefill_peak_caps["binding_limits"] == []


def test_full_8k_program_runs_when_prediction_fits_236_target(definitions):
    sched = decision(definitions, current=220, prediction=10.78)
    assert sched._adaptive_chunk_size(8192, request_id="fit", loop_label="cpu", kv_len=262144) == 8192
    assert sched._last_prefill_prediction["requested_chunk_tokens"] == 8192
    assert sched._last_prefill_prediction["executed_chunk_tokens"] == 8192
    assert sched._last_prefill_prediction["predicted_peak_bytes"] < 236 * GIB
    assert not sched._last_prefill_prediction["checkpoint_and_ssd_copies"]["included_in_prefill_transient"]


def test_genuine_low_headroom_retains_1024_fallback(definitions):
    sched = decision(definitions, hard=226.6, current=220.1, prediction=10.78)
    assert sched._adaptive_chunk_size(8192, request_id="low", loop_label="cpu", kv_len=262144) == 1024
    assert sched._last_prefill_peak_caps["effective_target_bytes"] < 236 * GIB
    assert sched._last_prefill_prediction["executed_chunk_tokens"] == 1024


def test_optional_upper_target_also_caps_atomic_chunk_guard_and_admission(definitions):
    sched = decision(definitions, current=220, prediction=17)
    assert sched._admission_limit_bytes() == 236 * GIB
    chosen = sched._guard_prefill_chunk(8192, kv_len=0, progress=0,
                                       loop_label="cpu", request_id="atomic")
    assert 32 <= chosen < 8192
    assert 220 * GIB + sched._predicted_chunk_transient(chosen) <= 236 * GIB
    control = decision(definitions, requested=None, current=220, prediction=17)
    assert control._guard_prefill_chunk(8192, kv_len=0, progress=0,
                                        loop_label="cpu", request_id="legacy") == 8192


def test_cache_storage_reports_logical_views_separately_from_growth_capacity(definitions):
    sched = decision(definitions)
    sched._last_prefill_prediction = {"request_id": "cache"}
    view = NS(nbytes=890 * 8192)
    buffer, idle = NS(nbytes=10 * 1024**2), NS(nbytes=10 * 1024**2)
    window, prime = NS(nbytes=128 * 528), NS(nbytes=1024)
    cache = NS(_dense=[None, window, view, None, None, None, None],
               _ds41_buffers={2: {"view": view, "buffer": buffer, "idle": idle}},
               _omlx_mtp_prime_ctx=NS(caches=[NS(keys=prime)]))
    sched._record_prefill_cache_storage([cache], "cache")
    metric = sched._last_prefill_prediction["completed_chunk_cache_storage"]
    assert metric["growing_kv_index_logical_bytes"] == 890 * 8192
    assert metric["registered_growing_kv_index_capacity_bytes"] == 20 * 1024**2
    assert metric["bounded_state_logical_bytes"] == 128 * 528
    assert metric["mtp_prime_logical_bytes"] == 1024
    assert metric["supported_cache_objects"] == 1


def test_native_descriptor_reaches_memory_profile_only_through_explicit_hook(definitions):
    descriptor = {"native": True}
    language = NS(_omlx_native_prefill_workspace_descriptor=lambda: descriptor)
    wrapper = NS(_vlm_model=NS(language_model=language))
    assert definitions.scheduler._deepseek_native_attention_descriptor(wrapper) is descriptor
    assert definitions.scheduler._deepseek_native_attention_descriptor(NS(config=descriptor)) is None
    def unavailable():
        raise RuntimeError("native binary capability unavailable")
    assert definitions.scheduler._deepseek_native_attention_descriptor(
        NS(_omlx_native_prefill_workspace_descriptor=unavailable)) is None
    cyclic = NS()
    cyclic.language_model = cyclic
    assert definitions.scheduler._deepseek_native_attention_descriptor(cyclic) is None


def test_runtime_factory_receives_native_proof_and_safety_multiplier_stays_13(definitions):
    scheduler = next(node for node in definitions.tree.body if isinstance(node, ast.ClassDef) and node.name == "Scheduler")
    factory_calls = [node for node in ast.walk(scheduler) if isinstance(node, ast.Call)
                     and isinstance(node.func, ast.Name) and node.func.id == "make_prefill_memory_profile"]
    assert factory_calls and all("deepseek_native_attention" in {kw.arg for kw in call.keywords}
                                 for call in factory_calls)
    safety = [node.value for node in scheduler.body if isinstance(node, ast.AnnAssign)
              and isinstance(node.target, ast.Name) and node.target.id == "_PREFILL_TRANSIENT_SAFETY"]
    assert len(safety) == 1 and ast.literal_eval(safety[0]) == 1.3
