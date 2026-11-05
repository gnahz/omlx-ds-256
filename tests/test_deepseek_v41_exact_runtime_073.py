"""Actual service wrapper/cache integration with CPU arrays and a fake engine."""
import asyncio
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS

import mlx.core as mx
import pytest

from omlx.patches.deepseek_v41.prefix_cache import ExactPrefixCache
from test_deepseek_v41_073_merge import pending_record, assert_identical

RUNTIME = Path(__file__).resolve().parents[1] / "deploy/deepseek"


def read_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


wrapper = read_module("exact_runtime073_test", RUNTIME / "omlx_v41_exact_ssd_runtime.py")


@pytest.fixture(autouse=True)
def cpu_arrays():
    with mx.stream(mx.cpu):
        yield


async def start_runtime(directory, fingerprint, *, signature=("dspark", 4)):
    manager = ExactPrefixCache(2**20)
    manager.signature = signature
    native_dir = directory / "native-ssd"
    settings = NS(base_path=directory, cache=NS(
        enabled=True, hot_cache_only=False, hot_cache_max_size="0",
        hot_cache_write_through=False, ssd_cache_dir=str(native_dir),
        ssd_cache_max_size="4MB",
        get_ssd_cache_dir=lambda _: native_dir,
        get_hot_cache_max_size_bytes=lambda: 0,
        get_ssd_cache_max_size_bytes=lambda _: 4 * 2**20,
    ))
    server = NS(_server_state=NS(global_settings=settings))
    runtime = wrapper.CacheRuntime(server, dict(
        enabled=True, capacity_gib=2 / 1024, persistence_enabled=True,
        directory=str(directory / "restart"), save_interval_seconds=300,
        shutdown_timeout_seconds=30, l2_enabled=True,
        native_ssd_cache_dir=str(native_dir)), fingerprint, directory / "receipt.json")
    guarded = []
    runtime.seed_guard = lambda scheduler: guarded.append(scheduler)
    async def on_engine(core, fn):
        assert mx.default_device() == mx.cpu
        return fn()
    async def periodic(core):
        await asyncio.Event().wait()
    runtime.on_engine, runtime.periodic = on_engine, periodic
    core = NS(config=NS(model_name="tiny"),
              scheduler=NS(_configure_exact_prefix_cache=lambda: manager))
    await runtime.initialize(core)
    assert guarded == [core.scheduler]
    assert runtime.models["tiny"]["state"] == "ready"
    return runtime, core, manager


@pytest.mark.asyncio
async def test_service_wrapper_roundtrips_ced_replay_and_separates_cache_tiers(tmp_path):
    assert mx.default_device() == mx.cpu
    fingerprint = {"runtime_sources": {"prefix_cache_store.py": "v3-hash"}}
    runtime, core, first = await start_runtime(tmp_path, fingerprint)
    tokens, before = pending_record(first)
    assert first._l2.directory != first._persistence.directory
    assert first._l2.identity == first._persistence.identity == {
        "deployment": fingerprint, "draft_signature": ("dspark", 4)}
    await runtime.flush(core)
    assert json.loads((tmp_path / "receipt.json").read_text())["saved"]
    assert not first.entries and core._service_exact_cache_manager is None
    reopened, core, second = await start_runtime(tmp_path, fingerprint)
    try:
        after, hit, _ = second.prepare(tokens)
        assert hit == 8192
        assert_identical(before, after)
        second.clear()
        after, hit, _ = second.prepare(tokens)
        assert hit == 8192 and second.snapshot_stats()["l2"]["hits"] == 1
        assert_identical(before, after)
    finally:
        await reopened.flush(core)


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["deployment", "draft"])
async def test_service_cache_identity_rejects_different_runtime_or_draft(tmp_path, change):
    before = {"runtime_sources": {"prefix_cache_store.py": "v2-hash"}}
    runtime, core, manager = await start_runtime(tmp_path, before)
    tokens, _ = pending_record(manager)
    await runtime.flush(core)
    fingerprint = ({"runtime_sources": {"prefix_cache_store.py": "v3-hash"}}
                   if change == "deployment" else before)
    signature = ("dspark", 3 if change == "draft" else 4)
    reopened, core, manager = await start_runtime(tmp_path, fingerprint, signature=signature)
    try:
        assert manager.prepare(tokens)[1] == 0
        assert manager.snapshot_stats()["persistence"]["restored_checkpoints"] == 0
        assert manager.snapshot_stats()["l2"]["invalidations"] == 1
    finally:
        await reopened.flush(core)


def test_launch_identity_tracks_payload_format_source_and_runtime_wrapper(tmp_path, monkeypatch):
    launch = read_module("launch073_identity_test", RUNTIME / "launch.py")
    root = tmp_path / "source"
    store = root / "omlx/patches/deepseek_v41/prefix_cache_store.py"
    store.parent.mkdir(parents=True)
    store.write_text('FORMAT = "v2"\n')
    (root / "apps/omlx-web/omlx_web").mkdir(parents=True)
    model = tmp_path / "fixture-model"
    model.mkdir()
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors.index.json"):
        (model / name).write_text('{}\n')
    native = tmp_path / "deepseek"
    native.mkdir()
    wrapper_file = native / "omlx_v41_exact_ssd_runtime.py"
    wrapper_file.write_text('# wrapper v1\n')
    proof = tmp_path / "proof.json"
    proof.write_text('{}\n')
    monkeypatch.setattr(launch, "HERE", native)
    profile = dict(source=str(root), target_commit="v073", metadata_proof=str(proof),
                   effective_settings={}, model_path=str(model))
    first = launch.runtime_identity(profile)
    store.write_text('FORMAT = "v3-ced-replay"\n')
    second = launch.runtime_identity(profile)
    assert first["runtime_sources"] != second["runtime_sources"]
    wrapper_file.write_text('# wrapper v2\n')
    third = launch.runtime_identity(profile)
    assert second["exact_ssd_runtime"] != third["exact_ssd_runtime"]


def test_migrated_service_paths_do_not_share_the_production_cache_namespace():
    profile = json.loads((RUNTIME / "profile.json").read_text())
    paths = [Path(profile["base_path"]), Path(profile["cache"]["directory"]),
             Path(profile["cache"]["native_ssd_cache_dir"])]
    assert all(path.is_relative_to(RUNTIME) for path in paths)
    assert len(set(paths)) == 3
