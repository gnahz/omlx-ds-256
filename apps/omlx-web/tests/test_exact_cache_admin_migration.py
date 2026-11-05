"""CPU-only regressions for the custom exact-cache dashboard API on 0.7.3."""

import asyncio
import copy
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from omlx.admin import routes
from omlx.settings import GlobalSettings


def setup_exact(monkeypatch, tmp_path):
    settings = GlobalSettings(base_path=tmp_path)
    settings.validate = lambda: []
    applied = []

    async def apply(value):
        applied.append(copy.deepcopy(value.cache))
        return True, "Exact cache settings applied"

    exact = SimpleNamespace(
        native_settings_lock=asyncio.Lock(),
        validate_native_settings_request=lambda request, value: None,
        apply_native_settings=apply,
        clear_native_cache=AsyncMock(return_value={"success": True}),
        ready=True,
        error=None,
    )
    state = SimpleNamespace(exact_prefix_cache_runtime=exact)
    # The route only needs service state. Importing a serving engine is not
    # necessary for this settings/API test and must never load a model.
    server = ModuleType("omlx.server")
    server._server_state = state
    monkeypatch.setitem(sys.modules, "omlx.server", server)
    monkeypatch.setattr(routes, "_get_server_state", lambda: state)
    monkeypatch.setattr(routes, "_get_global_settings", lambda: settings)
    save = AsyncMock()
    monkeypatch.setattr(routes, "_save_global_settings_async", save)
    return settings, exact, applied, save


def test_exact_cache_settings_use_new_runtime_signature(monkeypatch, tmp_path):
    settings, exact, applied, save = setup_exact(monkeypatch, tmp_path)
    request = routes.GlobalSettingsRequest(ssd_cache_max_size="60GB")
    result = asyncio.run(routes.update_global_settings(request, is_admin=True))
    assert result["success"] is True
    assert result["runtime_applied"] == ["cache"]
    assert settings.cache.ssd_cache_max_size == "60GB"
    assert [cache.ssd_cache_max_size for cache in applied] == ["60GB"]
    save.assert_awaited_once_with(settings)


def test_exact_cache_settings_restore_runtime_when_save_fails(monkeypatch, tmp_path):
    settings, exact, applied, save = setup_exact(monkeypatch, tmp_path)
    original = copy.deepcopy(settings.cache)
    save.side_effect = OSError("simulated settings storage failure")
    request = routes.GlobalSettingsRequest(ssd_cache_max_size="60GB")
    with pytest.raises(routes.HTTPException) as failure:
        asyncio.run(routes.update_global_settings(request, is_admin=True))
    assert failure.value.status_code == 500
    assert settings.cache == original
    assert [cache.ssd_cache_max_size for cache in applied] == [
        "60GB", original.ssd_cache_max_size,
    ]
    assert exact.ready is True


@pytest.mark.parametrize("ssd", [False, True])
def test_exact_cache_clear_keeps_tier_selection(monkeypatch, tmp_path, ssd):
    _, exact, _, _ = setup_exact(monkeypatch, tmp_path)
    action = routes.clear_ssd_cache if ssd else routes.clear_hot_cache
    assert asyncio.run(action(is_admin=True)) == {"success": True}
    exact.clear_native_cache.assert_awaited_once_with(ssd=ssd)


def test_exact_cache_observability_keeps_checkpoints_and_tier_counters():
    exact = {
        "bytes": 1024, "max_bytes": 2048, "checkpoints": 7,
        "hits": 11, "ram_hits": 8, "l2_hits": 3, "evictions": 2,
        "l2": {"entries": 9, "disk_bytes": 4096, "max_bytes": 8192,
               "hits": 3, "misses": 4, "reads": 5, "writes": 6,
               "read_bytes": 700, "write_bytes": 800, "pending_bytes": 900},
    }
    row = routes._exact_cache_observability(exact)
    assert row["cache_tier"] == "deepseek-exact-l2"
    assert row["exact_prefix_cache"]["checkpoints"] == 7
    assert (row["hot_cache_entries"], row["num_files"]) == (7, 9)
    assert (row["read_bytes"], row["write_bytes"], row["pending_bytes"]) == (700, 800, 900)
    assert row["cache_rates"]["cumulative"]["ssd_hot_hits"] == 8
    assert row["cache_rates"]["cumulative"]["ssd_disk_loads"] == 3
