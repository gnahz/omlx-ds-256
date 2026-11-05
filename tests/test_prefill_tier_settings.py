"""CPU-only settings and actual enforcer propagation for prefill tiers."""

from __future__ import annotations

import ast
import asyncio
from collections import deque
import importlib.util
import json
import logging
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import pytest

from omlx.settings import GlobalSettings, MemorySettings


SOURCE = Path(__file__).resolve().parents[1]
GIB = 1024**3


@pytest.fixture(autouse=True)
def cpu_network_validation(monkeypatch):
    # settings.validate needs the pure network helpers; importing their
    # package initializer also loads hardware/MLX. Load the real helpers
    # directly when they are not already present in a larger test session.
    name = "omlx.utils.network"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, SOURCE / "omlx/utils/network.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        monkeypatch.setitem(sys.modules, name, module)


def _tier_errors(memory):
    settings = GlobalSettings()
    settings.memory = memory
    return [error for error in settings.validate() if "prefill_" in error]


@pytest.mark.parametrize("minimum,preferred", [(256, 512), (256, 256), (1024, 8192)])
def test_prefill_tier_configuration_roundtrip(minimum, preferred):
    memory = MemorySettings.from_dict({
        "prefill_min_chunk_tokens": minimum,
        "prefill_preferred_chunk_tokens": preferred,
    })
    assert memory.prefill_min_chunk_tokens == minimum
    assert memory.prefill_preferred_chunk_tokens == preferred
    assert MemorySettings.from_dict(memory.to_dict()) == memory
    assert _tier_errors(memory) == []


@pytest.mark.parametrize("values", [{}, {"prefill_min_chunk_tokens": 256}])
def test_legacy_settings_keep_new_policy_disabled(values):
    memory = MemorySettings.from_dict(values)
    assert memory.prefill_preferred_chunk_tokens == 0
    assert MemorySettings.from_dict(memory.to_dict()).prefill_preferred_chunk_tokens == 0
    assert _tier_errors(memory) == []


@pytest.mark.parametrize("preferred", [-1, 1, 255, 8193, True, 512.0])
def test_invalid_preferred_floor_is_rejected(preferred):
    errors = _tier_errors(MemorySettings(
        prefill_min_chunk_tokens=256, prefill_preferred_chunk_tokens=preferred,
    ))
    assert len(errors) == 1
    assert "prefill_preferred_chunk_tokens" in errors[0]


@pytest.mark.parametrize("minimum", [0, 1025])
def test_emergency_floor_keeps_existing_bounds(minimum):
    errors = _tier_errors(MemorySettings(
        prefill_min_chunk_tokens=minimum, prefill_preferred_chunk_tokens=0,
    ))
    assert len(errors) == 1
    assert "prefill_min_chunk_tokens" in errors[0]


@pytest.fixture
def tier_settings_file(tmp_path, monkeypatch):
    monkeypatch.delenv("OMLX_PREFILL_MIN_CHUNK_TOKENS", raising=False)
    monkeypatch.delenv("OMLX_PREFILL_PREFERRED_CHUNK_TOKENS", raising=False)
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"memory": {
        "prefill_min_chunk_tokens": 32, "prefill_preferred_chunk_tokens": 0,
    }}))
    return path


def test_load_pinned_tiers_overrides_legacy_launcher_file(tier_settings_file, monkeypatch):
    monkeypatch.setenv("OMLX_PREFILL_MIN_CHUNK_TOKENS", "256")
    monkeypatch.setenv("OMLX_PREFILL_PREFERRED_CHUNK_TOKENS", "512")
    # Every load starts from the same legacy file, just like prepare_settings
    # writing the baseline on each start; the registered env must win each time.
    for _ in range(2):
        settings = GlobalSettings.load(base_path=tier_settings_file.parent)
        assert settings.memory.prefill_min_chunk_tokens == 256
        assert settings.memory.prefill_preferred_chunk_tokens == 512
        assert _tier_errors(settings.memory) == []
        assert json.loads(tier_settings_file.read_text())["memory"]["prefill_min_chunk_tokens"] == 32


@pytest.mark.parametrize("minimum,preferred", [(32, 0), (256, 1024)])
def test_load_without_tier_environment_preserves_file_values(tier_settings_file, minimum, preferred):
    tier_settings_file.write_text(json.dumps({"memory": {
        "prefill_min_chunk_tokens": minimum, "prefill_preferred_chunk_tokens": preferred,
    }}))
    settings = GlobalSettings.load(base_path=tier_settings_file.parent)
    assert settings.memory.prefill_min_chunk_tokens == minimum
    assert settings.memory.prefill_preferred_chunk_tokens == preferred


@pytest.mark.parametrize("name", ["OMLX_PREFILL_MIN_CHUNK_TOKENS", "OMLX_PREFILL_PREFERRED_CHUNK_TOKENS"])
@pytest.mark.parametrize("value", ["bad", "512.0", ""])
def test_noninteger_tier_environment_stops_load(tier_settings_file, monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=name + " must be an integer"):
        GlobalSettings.load(base_path=tier_settings_file.parent)


@pytest.mark.parametrize("minimum,preferred", [(0, 512), (256, 255), (256, 8193)])
def test_integer_environment_bounds_are_checked_by_validation(tier_settings_file, monkeypatch, minimum, preferred):
    monkeypatch.setenv("OMLX_PREFILL_MIN_CHUNK_TOKENS", str(minimum))
    monkeypatch.setenv("OMLX_PREFILL_PREFERRED_CHUNK_TOKENS", str(preferred))
    settings = GlobalSettings.load(base_path=tier_settings_file.parent)
    assert _tier_errors(settings.memory)


def _host_enforcer_class():
    """Execute the actual class without importing any GPU/engine modules.

    The constructor and propagation method are unmodified production code;
    hardware probes are replaced on each instance before propagation.
    """
    path = SOURCE / "omlx/process_memory_enforcer.py"
    parsed = ast.parse(path.read_text())
    constants = {
        "_TIER_RESERVE", "_VALID_TIERS", "_LEGACY_SOFT_THRESHOLD",
        "_SOFT_THRESHOLD_BY_TIER", "_LEGACY_HARD_THRESHOLD",
        "_HARD_THRESHOLD_BY_TIER", "_PREFILL_HEADROOM_SAFETY",
        "_PREFILL_ABORT_MARGIN",
    }
    nodes = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
    for node in parsed.body:
        names = ([target.id for target in node.targets if isinstance(target, ast.Name)]
                 if isinstance(node, ast.Assign) else
                 [node.target.id] if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
                 else [])
        if constants.intersection(names) or isinstance(node, ast.ClassDef) and node.name == "ProcessMemoryEnforcer":
            nodes.append(node)
    namespace = {"deque": deque, "asyncio": asyncio, "logger": logging.getLogger(__name__)}
    tree = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace["ProcessMemoryEnforcer"]


@pytest.mark.parametrize("preferred", [0, 512, 1024])
def test_actual_enforcer_propagates_both_floors_without_changing_limits(preferred):
    before = set(sys.modules)
    direct = NS(batch_generator=NS(_memory_limit_bytes=0))
    wrapped = NS(batch_generator=NS(_memory_limit_bytes=0))
    # A disabled policy must overwrite any stale preferred floor too.
    direct._prefill_preferred_chunk_tokens = 8192
    pool = NS(_entries={
        "direct": NS(engine=NS(scheduler=direct)),
        "wrapped": NS(engine=NS(_engine=NS(engine=NS(scheduler=wrapped)))),
        "unloaded": NS(engine=None),
    })
    enforcer = _host_enforcer_class()(
        pool, memory_guard_tier="custom", memory_guard_custom_ceiling_gb=240,
        soft_threshold=.97, hard_threshold=.999,
        prefill_min_chunk_tokens=256, prefill_preferred_chunk_tokens=preferred,
    )
    enforcer._get_ceiling_breakdown = lambda: {
        "static": 240 * GIB, "dynamic": 240 * GIB,
        "metal_cap": 240 * GIB, "hard_limit": 240 * GIB,
    }
    enforcer._get_effective_metal_cap_bytes = lambda: 240 * GIB
    enforcer._get_abort_limit_bytes = lambda: 240 * GIB
    enforcer._hot_cache_reserved_bytes = lambda: 0
    enforcer._scheduler_limit_bytes = lambda value, **kwargs: value
    enforcer._propagate_memory_limit()
    for scheduler in (direct, wrapped):
        assert scheduler._prefill_min_chunk_tokens == 256
        assert scheduler._prefill_preferred_chunk_tokens == preferred
        assert scheduler._memory_hard_limit_bytes == 240 * GIB
        assert scheduler._memory_abort_limit_bytes == 240 * GIB
        assert scheduler._memory_limit_bytes == int(240 * GIB * .97)
        assert scheduler._memory_hard_watermark_bytes == int(240 * GIB * .999)
        assert scheduler._prefill_memory_guard is True
    assert not {"mlx.core", "omlx.engine_pool"}.intersection(set(sys.modules) - before)


def test_server_wires_memory_settings_to_enforcer_constructor():
    path = SOURCE / "omlx/server.py"
    tree = ast.parse(path.read_text())
    calls = [node for node in ast.walk(tree)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
             and node.func.id == "ProcessMemoryEnforcer"]
    assert len(calls) == 1
    settings = GlobalSettings()
    settings.memory = MemorySettings(
        prefill_min_chunk_tokens=256, prefill_preferred_chunk_tokens=512,
    )
    state = NS(global_settings=settings, engine_pool=object(), settings_manager=object())
    args = eval(compile(ast.Expression(body=calls[0]), str(path), "eval"), {
        "ProcessMemoryEnforcer": lambda **kwargs: kwargs,
        "_server_state": state, "memory_settings": settings.memory,
    })
    assert args["prefill_min_chunk_tokens"] == 256
    assert args["prefill_preferred_chunk_tokens"] == 512
    assert args["engine_pool"] is state.engine_pool
    assert args["global_settings"] is settings
