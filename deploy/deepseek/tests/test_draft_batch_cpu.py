"""CPU-only deployment checks; no model imports, service access, or research files."""
from __future__ import annotations

import ast
import builtins
import copy
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest


DEPLOY = Path(__file__).resolve().parents[1]
LAUNCH = DEPLOY / "launch.py"


def function(name):
    return next(node for node in ast.parse(LAUNCH.read_text()).body
                if isinstance(node, ast.FunctionDef) and node.name == name)


def execute(nodes, scope):
    # Extract only the behavior under test: importing the inference launcher or
    # invoking serve would make this a model/service test instead of a CPU test.
    def forbid_import(*args, **kwargs):
        pytest.fail("Deployment snapshot/guard must not import any module")

    scope["__builtins__"] = dict(vars(builtins), __import__=forbid_import)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(LAUNCH), "exec"), scope)
    return scope


def test_template_enables_draft_batch_and_preserves_serving_budgets():
    profile = json.loads((DEPLOY / "profile.template.json").read_text())
    settings = profile["effective_settings"]
    assert settings["dspark_draft_batch"] is True
    assert profile["environment"]["DS41_DRAFT_BATCH"] == "1"
    assert settings["memory_guard_gb"] == profile["process_limit_gib"] == 240
    assert settings["prefill_peak_target_gib"] == 238
    assert settings["wired_limit_mib"] == 245760
    assert settings["max_concurrent_requests"] == 4
    assert settings["prefill_pool_bytes"] == 3 * 1024**3
    assert settings["prefill_whole_reservation"] is True
    assert settings["mtp_adaptive_max_depth"] == 4
    assert settings["mtp_fixed_depth"] is None
    assert settings["temperature"] == settings["top_p"] == 1.0
    assert profile["cache"]["capacity_gib"] == 2
    assert profile["cache"]["ssd_cache_max_size"] == "60GB"
    assert profile["memory_policy"]["system_budget_gib"] == 16
    assert profile["host"] == "127.0.0.1"
    assert profile["port"] == 8080


def startup_guard(enabled, declared):
    nodes = function("serve").body
    guard = next(node for node in nodes if isinstance(node, ast.If)
                 and "CANDIDATE_DRAFT_BATCH_ENABLED" in ast.unparse(node.test))
    # Reject a mismatch before configuring/starting the engine.
    initialize = next(node for node in nodes if isinstance(node, ast.FunctionDef)
                      and node.name == "initialize")
    assert nodes.index(guard) < nodes.index(initialize)
    settings = {} if declared is None else {"dspark_draft_batch": declared}
    execute([guard], {"dspark": NS(CANDIDATE_DRAFT_BATCH_ENABLED=enabled),
                      "profile": {"effective_settings": settings}})


@pytest.mark.parametrize("enabled,declared", [(False, False), (True, True), (False, None)])
def test_startup_accepts_matching_actual_and_declared_switch(enabled, declared):
    startup_guard(enabled, declared)


@pytest.mark.parametrize("enabled,declared", [(False, True), (True, False), (True, None)])
def test_startup_rejects_mismatch_or_undeclared_activation(enabled, declared):
    with pytest.raises(RuntimeError, match="declared service profile"):
        startup_guard(enabled, declared)


@pytest.mark.parametrize("enabled", [False, True])
def test_snapshot_reads_actual_state_without_imports_or_counter_reset(enabled):
    metrics = {"counts": {"shared_forwards": 2}, "groups": {"2x4": 2}, "fallbacks": {}}
    before = copy.deepcopy(metrics)
    resets = []

    def stats(reset=False):
        resets.append(reset)
        return copy.deepcopy(metrics)

    draft = NS(CANDIDATE_DRAFT_BATCH_ENABLED=enabled, CANDIDATE_DIAGNOSTICS_ENABLED=False,
               _MAX_BATCH_ROWS=8, candidate_draft_stats=stats)
    native = NS(compact_expert=NS())
    modules = {}
    scope = execute([function("draft_runtime_actual")],
                    {"sys": NS(modules=modules), "os": NS(environ={})})
    read = scope["draft_runtime_actual"]
    actual = read(draft, native)
    assert actual["draft_batch"] == dict(enabled=enabled, diagnostics_enabled=False,
                                        max_rows=8, stats=metrics)
    assert actual["experiment_controls"] == dict(attention=False, grouped=False,
        batched_verify=True, batch_projections=True, loaded_shims=[])
    assert modules == {}
    assert resets == [False]
    assert metrics == before
    # Consumers can inspect or alter a returned snapshot without resetting stats.
    actual["draft_batch"]["stats"]["counts"]["shared_forwards"] = 0
    assert read(draft, native)["draft_batch"]["stats"] == before
    assert resets == [False, False]


def test_snapshot_reports_already_loaded_controls_without_loading_shims():
    modules = {"omlx.patches.deepseek_v41.batched_verify": NS(CANDIDATE_ATTENTION_ENABLED=True),
               "grouped_metadata": NS()}
    draft = NS(CANDIDATE_DRAFT_BATCH_ENABLED=True, CANDIDATE_DIAGNOSTICS_ENABLED=True,
               _MAX_BATCH_ROWS=8, candidate_draft_stats=lambda: {})
    native = NS(compact_expert=NS(_concurrency_grouped_metadata_controller=NS(enabled=True)))
    scope = execute([function("draft_runtime_actual")],
                    {"sys": NS(modules=modules), "os": NS(environ={
                        "DS41_BATCH_VERIFY": "0", "DS41_BATCH_PROJECTIONS": "0"})})
    read = scope["draft_runtime_actual"]
    before = dict(modules)
    actual = read(draft, native)
    assert actual["draft_batch"]["diagnostics_enabled"] is True
    assert actual["experiment_controls"] == dict(attention=True, grouped=True,
        batched_verify=False, batch_projections=False, loaded_shims=["grouped_metadata"])
    assert modules == before
    modules["omlx.patches.deepseek_v41.batched_verify"].CANDIDATE_ATTENTION_ENABLED = False
    modules["omlx.patches.deepseek_v41.attention_service"] = NS(_installed=NS(enabled=True))
    assert read(draft, native)["experiment_controls"]["attention"] is True


def test_runtime_actual_includes_draft_snapshot_in_existing_diagnostics():
    update = next(node for node in function("runtime_actual").body
                  if isinstance(node, ast.Expr)
                  and "draft_runtime_actual(" in ast.unparse(node))
    draft, native = object(), object()
    snapshot = {"draft_batch": {"enabled": True, "max_rows": 8}}

    def read(actual_draft, actual_native):
        assert actual_draft is draft and actual_native is native
        return snapshot

    actual = {"dspark_draft_async": True}
    execute([update], {"actual": actual, "dspark": draft, "native_metadata": native,
                       "draft_runtime_actual": read})
    assert actual == {"dspark_draft_async": True, **snapshot}


def test_snapshot_has_no_imports_gpu_calls_or_research_dependency():
    node = function("draft_runtime_actual")
    assert not any(isinstance(child, (ast.Import, ast.ImportFrom)) for child in ast.walk(node))
    assert "research/" not in ast.unparse(node)
    # The one delegated call is the Python statistics reader. In particular,
    # this helper must never evaluate tensors or synchronize a device.
    allowed = {"sys.modules.get", "getattr", "bool", "dspark.candidate_draft_stats", "os.environ.get"}
    assert all(ast.unparse(child.func) in allowed for child in ast.walk(node)
               if isinstance(child, ast.Call))
