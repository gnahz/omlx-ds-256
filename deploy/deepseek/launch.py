#!/usr/bin/env python3
"""Serve the portable DeepSeek profile in an independent process."""
from __future__ import annotations

import argparse
import copy
import functools
import hashlib
import json
import os
from pathlib import Path
import sys
import threading

HERE = Path(__file__).resolve().parent
from portable import REPO, state_dir, profile_path
PROJECT = REPO
PROFILE = profile_path()


def read_profile():
    value = json.loads(PROFILE.read_text())
    required = {"source", "python", "model_path", "model_id", "base_path", "api_key_file", "metadata_proof", "cache", "effective_settings"}
    if not required <= value.keys():
        raise ValueError("Migration profile lacks required fields: " + str(sorted(required - value.keys())))
    validate_web_assets(value)
    return value


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_web_assets(profile):
    """The accepted profile pins the actual external web package's assets."""
    root = (Path(profile["source"]) / "apps/omlx-web/omlx_web").resolve()
    references = profile.get("web_asset_refs")
    if not references:
        raise ValueError("Exact-cache service profile lacks web asset identities")
    for name, expected in references.items():
        target = (root / name).resolve()
        if not target.is_relative_to(root) or sha(target) != expected:
            raise RuntimeError(f"The accepted dashboard asset differs: {name}")


def runtime_identity(profile):
    source = Path(profile["source"])
    suffixes = {".py", ".so", ".dylib", ".metallib", ".metal", ".cpp", ".h", ".hpp"}
    files = {str(path.relative_to(source)): sha(path)
             for path in sorted((source / "omlx").rglob("*"))
             if path.is_file() and path.suffix in suffixes}
    web = source / "apps/omlx-web/omlx_web"
    web_files = {str(path.relative_to(source)): sha(path) for path in sorted(web.rglob("*"))
                 if path.is_file() and path.suffix in {".py", ".html", ".js", ".css", ".json"}}
    return {"target_commit": profile["target_commit"], "runtime_sources": files,
            "web_assets": web_files, "exact_ssd_runtime": sha(HERE / "omlx_v41_exact_ssd_runtime.py"),
            "launcher": sha(__file__), "metadata_proof": sha(profile["metadata_proof"]),
            "effective_settings": profile["effective_settings"], "model": profile["model_path"],
            "cold_prefix_baseline": profile.get("cold_prefix_baseline", False),
            "model_files": {name: sha(Path(profile["model_path"]) / name) for name in
                            ("config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors.index.json")}}


def owned_native_cache_settings(profile, current, marker):
    """Seed once, then preserve dashboard-edited SSD settings across restarts."""
    from omlx.config import parse_size
    native_dir = str(Path(profile["cache"]["native_ssd_cache_dir"]).resolve())
    expected = {"schema_version": 1, "backend": "deepseek_v41_exact_l2_v1",
                "directory": native_dir, "ram_capacity_gib": profile["cache"]["capacity_gib"]}
    cache = dict(current or {})
    if marker is None:
        cache.update(enabled=True, ssd_cache_dir=native_dir,
                     ssd_cache_max_size=profile["cache"]["ssd_cache_max_size"],
                     hot_cache_only=False, hot_cache_max_size="0", hot_cache_write_through=False)
    else:
        if marker != expected:
            raise RuntimeError("Native exact-cache settings ownership differs from the service profile")
        if Path(cache.get("ssd_cache_dir", "")).expanduser().resolve() != Path(native_dir):
            raise RuntimeError("Native exact-cache directory changed outside the service profile")
        if parse_size(str(cache.get("hot_cache_max_size", "0"))) != 0 or cache.get("hot_cache_write_through", False):
            raise RuntimeError("Native paged hot-cache settings cannot override the exact-cache RAM profile")
    limit = str(cache["ssd_cache_max_size"])
    if limit.lower() != "auto" and parse_size(limit) <= 0:
        raise ValueError("Native SSD cache size must be positive")
    return cache, expected


def prepare_settings(profile, host, port):
    """Copy the existing model policy into a fresh private service namespace."""
    base = Path(profile["base_path"])
    base.mkdir(mode=0o700, parents=True, exist_ok=True)
    registry = Path(profile["state_path"]) / "model-registry"
    registry.mkdir(exist_ok=True)
    model = registry / profile["model_id"]
    if model.exists() or model.is_symlink():
        if model.resolve() != Path(profile["model_path"]).resolve():
            raise RuntimeError("Migration model registry points to another checkpoint")
    else:
        model.symlink_to(profile["model_path"], target_is_directory=True)
    settings = json.loads((HERE / "settings.template.json").read_text())
    settings["server"].update(host=host, port=port)
    settings["model"].update(model_dirs=[str(registry)], model_dir=str(registry))
    policy_file = base / "exact-l2-settings-policy.json"
    settings_file = base / "settings.json"
    marker = json.loads(policy_file.read_text()) if policy_file.exists() else None
    previous = json.loads(settings_file.read_text()).get("cache", {}) if marker is not None else settings["cache"]
    settings["cache"], native_policy = owned_native_cache_settings(profile, previous, marker)
    settings["logging"]["log_dir"] = str(Path(profile["state_path"]) / "logs")
    settings["auth"].update(api_key=Path(profile["api_key_file"]).read_text().strip(),
                            skip_api_key_verification=False, allow_unauthenticated_inference=False)
    for name, value in (("settings.json", settings), ("model_settings.json", profile["model_settings"])):
        path = base / name
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(value, indent=2) + "\n")
        temporary.replace(path)
        path.chmod(0o600)
    temporary = policy_file.with_suffix(".tmp")
    temporary.write_text(json.dumps(native_policy, indent=2) + "\n")
    temporary.replace(policy_file)
    policy_file.chmod(0o600)
    return registry


def require_native_single_decode(attention_fast, enabled):
    """Reject a stale native binary before loading the production checkpoint."""
    if enabled and not attention_fast.has_symbol("packed_sparse_attention_single_decode"):
        raise RuntimeError("Required native singleton packed attention capability is unavailable")


def count_runtime_call(module, name, label, *, stats, lock):
    """Record the admitted call before delegation, including the first call."""
    original = getattr(module, name)
    stats.setdefault(label, 0)
    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        with lock:
            stats[label] += 1
        return original(*args, **kwargs)
    setattr(module, name, wrapped)


def runtime_actual(server, model_id):
    """Read the loaded engine's policy and module switches without evaluating weights."""
    from omlx.patches.deepseek_v41 import dspark, growth, native_metadata, native_metadata_prefill, storage
    actual = {"dspark_draft_async": bool(dspark.DRAFT_ASYNC),
              "native_metadata": native_metadata.enabled_proof() is not None,
              "native_metadata_prefill": bool(native_metadata_prefill.ENABLED),
              "growth": bool(growth.ENABLED), "pingpong": bool(growth.PINGPONG),
              "engram_storage": storage.runtime_status()}
    from omlx.custom_kernels.attention import fast as attention_fast
    actual["native_single_decode_capability"] = attention_fast.has_symbol("packed_sparse_attention_single_decode")
    from omlx.patches.deepseek_v41 import language, kernels, hc_fuse, attn_fusions, attn_in
    from omlx.patches.deepseek_v41 import fast_qmv, woa_compact, output_parser
    from omlx.patches.mlx_lm_mtp import copy_draft, batch_generator
    actual.update(decode_async_layers=language.DS41_DECODE_ASYNC,
                  prefill_pool_bytes=language.DS41_PREFILL_POOL,
                  prefill_pipeline=bool(language.DS41_PREFILL_PIPELINE),
                  short_row_controls={
                      "mhc": bool(language.DS41_MHC),
                      "hc_fuse": bool(hc_fuse.DS41_HC_FUSE),
                      "hc_early_submit": bool(hc_fuse.EARLY_SUBMIT),
                      "attention_fuse": bool(attn_fusions.ENABLED),
                      "attention_input_fuse": bool(attn_in.ENABLED),
                      "fast_qmv": bool(fast_qmv.ENABLED),
                      "woa_compact": bool(woa_compact.ENABLED),
                      "index_rows": bool(kernels.DS41_INDEX_ROWS),
                      "index_select": bool(attn_fusions.DECODE_SELECT),
                      "decode_single_tile": bool(kernels.DS41_DECODE_SINGLE_TILE),
                      "decode_radix": bool(kernels.DS41_DECODE_RADIX),
                      "prefill_mid_index": bool(kernels.DS41_PREFILL_MID_INDEX),
                      "prefill_index_legacy_mode": kernels.DS41_PREFILL_INDEX,
                      "prefill_index_implementation": kernels.PREFILL_INDEX_IMPLEMENTATION,
                      "batch_projections": os.environ.get("DS41_BATCH_PROJECTIONS", "1") != "0",
                      "detokenizer_reuse": bool(output_parser.REUSE)},
                  copy_draft={"enabled": bool(copy_draft.ENABLED), "ngram": copy_draft.NGRAM,
                              "min": copy_draft.MIN_DRAFT, "max": copy_draft.MAX_DRAFT,
                              "max_match": copy_draft.MAX_MATCH,
                              "generic_context_copy_for_dspark":
                                  bool(batch_generator.DSPARK_GENERIC_CONTEXT_COPY)})
    pool = server._server_state.engine_pool
    entry = None if pool is None else pool._entries.get(model_id)
    scheduler = (None if entry is None or entry.engine is None
                 else pool._resolve_scheduler_from_engine(entry.engine))
    if scheduler is not None:
        config = scheduler.config
        actual.update(max_concurrent_requests=config.max_num_seqs,
                      prefill_step_size=config.prefill_step_size,
                      prefill_peak_target_gib=getattr(config, "prefill_peak_target_gib", None),
                      preserved_prefill_max_defer_seconds=config.preserved_prefill_max_defer_seconds,
                      prefill_speed_priority=config.prefill_speed_priority,
                      running_requests=len(scheduler.running),
                      memory_guard_tier=getattr(scheduler, "_memory_guard_tier", None),
                      prefill_memory_guard=bool(getattr(scheduler, "_prefill_memory_guard", False)),
                      memory_static_ceiling_bytes=getattr(scheduler, "_memory_static_ceiling_bytes", 0),
                      memory_hard_limit_bytes=getattr(scheduler, "_memory_hard_limit_bytes", 0))
        if hasattr(scheduler, "_prefill_peak_target_caps"):
            actual["prefill_limits"] = scheduler._prefill_peak_target_caps()
        actual["last_prefill_prediction"] = getattr(scheduler, "_last_prefill_prediction", None)
        model = getattr(scheduler, "model", None)
        language = getattr(model, "_language_model", None)
        if language is None:
            language = getattr(model, "language_model", None)
        if language is None:
            language = getattr(getattr(model, "_vlm_model", None), "language_model", model)
        actual.update(mtp_runtime_enabled=bool(getattr(language, "_omlx_mtp_decode_enabled", False)),
                      mtp_runtime_depth=getattr(language, "_omlx_mtp_depth", None))
        head = getattr(language, "head", None)
        if head is not None and hasattr(head, "weight"):
            actual["vocabulary_head"] = {
                "weight_dtype": str(head.weight.dtype), "weight_shape": list(head.weight.shape),
                "bits": getattr(head, "bits", None), "mode": getattr(head, "mode", None),
                "group_size": getattr(head, "group_size", None),
                "quantize_input": getattr(head, "quantize_input", False)}
    enforcer = None if pool is None else getattr(pool, "_process_memory_enforcer", None)
    if enforcer is not None:
        actual["memory_guard_custom_ceiling_bytes"] = enforcer.memory_guard_custom_ceiling_bytes
    return actual


def candidate_cache_runtime(base):
    """Seed stable guard fields before an unpublished engine restores its cache."""
    class CandidateCacheRuntime(base):
        def seed_guard(self, scheduler):
            super().seed_guard(scheduler)
            enforcer = self.server._server_state.process_memory_enforcer
            breakdown = enforcer._get_ceiling_breakdown()
            scheduler._memory_static_ceiling_bytes = breakdown["static"]
            scheduler._memory_dynamic_ceiling_bytes = breakdown["dynamic"]
            scheduler._memory_metal_cap_bytes = breakdown["metal_cap"]
            scheduler._memory_metal_cap_raw_bytes = enforcer._get_effective_metal_cap_bytes()
            scheduler._memory_hot_cache_reserved_bytes = enforcer._hot_cache_reserved_bytes()
            scheduler._memory_guard_tier = enforcer._memory_guard_tier
            scheduler._prefill_memory_guard = enforcer._prefill_memory_guard
            scheduler._admission_paused = enforcer._pressure_level != "ok"
            scheduler._prefill_headroom_safety = enforcer._prefill_headroom_safety
            scheduler._prefill_safe_zone_ratio = enforcer._prefill_safe_zone_ratio
            scheduler._prefill_min_chunk_tokens = enforcer._prefill_min_chunk_tokens
            batch = getattr(scheduler, "batch_generator", None)
            if batch is not None and hasattr(batch, "_memory_limit_bytes"):
                batch._memory_limit_bytes = scheduler._memory_limit_bytes
                batch._memory_hard_limit_bytes = scheduler._memory_hard_limit_bytes

    return CandidateCacheRuntime


def qualification_cache_profile(profile, *, qualification, cold_prefix_baseline):
    if not cold_prefix_baseline:
        return profile
    if not qualification:
        raise ValueError("--cold-prefix-baseline requires --qualification")
    result = copy.deepcopy(profile)
    result["cold_prefix_baseline"] = True
    result["cache"]["persistence_enabled"] = False
    result["cache"]["l2_enabled"] = False
    result["cache"]["directory"] += "-cold-baseline"
    return result


def serve(profile, host, port, *, qualification=False, cold_prefix_baseline=False):
    validate_web_assets(profile)
    from portable import validate_configuration
    validate_configuration(profile)
    from omlx_physical_memory_guard import wait_for_startup_memory, install_native_load_guard, install_runtime_guard
    memory_admission = wait_for_startup_memory(profile)
    print("Physical memory admission: " + json.dumps(memory_admission, sort_keys=True), flush=True)
    profile = qualification_cache_profile(profile, qualification=qualification,
                                          cold_prefix_baseline=cold_prefix_baseline)
    sys.path.insert(0, profile["source"])
    registry = prepare_settings(profile, host, port)
    for name in list(os.environ):
        if (name.startswith(("DS41_", "DS4_", "BASELINE_", "LOCAL_MOE_", "OMLX_"))
                and name not in {"OMLX_DS_STATE", "OMLX_DS_INSTANCE_TOKEN"}):
            os.environ.pop(name)
    os.environ.update(profile["environment"])
    for name in ("OMLX_SSD_CACHE_DIR", "OMLX_SSD_CACHE_MAX_SIZE", "OMLX_HOT_CACHE_ONLY", "OMLX_HOT_CACHE_WRITE_THROUGH"):
        os.environ.pop(name, None)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      HF_HOME=str(Path(profile["state_path"]) / "cache/huggingface"),
                      XDG_CACHE_HOME=str(Path(profile["state_path"]) / "cache/xdg"))
    sys.path.insert(0, str(HERE))
    sys.path.insert(0, profile["source"])
    web_root = Path(profile["source"]) / "apps/omlx-web"
    sys.path.insert(0, str(web_root))
    import importlib.util
    web_spec = importlib.util.find_spec("omlx_web")
    if web_spec is None or not Path(web_spec.origin).resolve().is_relative_to(web_root.resolve()):
        raise RuntimeError("The service must serve its fingerprinted omlx_web assets")
    from fastapi import Depends, HTTPException
    from omlx import server, process_memory_enforcer as memory_policy
    physical_guard = install_runtime_guard(memory_policy.ProcessMemoryEnforcer, profile)
    from omlx.cli import main as cli_main
    from omlx.scheduler import SchedulerConfig
    from omlx.patches.deepseek_v41 import dspark, growth, native_metadata, native_metadata_prefill
    from omlx.engine_pool import EnginePool
    native_load_guard = install_native_load_guard(native_metadata.NativeMetadataPlan, EnginePool, profile)
    sys.path.insert(0, str(HERE))
    from omlx_v41_exact_ssd_runtime import CacheRuntime

    from omlx.custom_kernels.attention import fast as attention_fast
    from omlx.patches.deepseek_v41 import language
    require_native_single_decode(attention_fast, language.DS41_NATIVE_DECODE)

    if not (dspark.DRAFT_ASYNC and native_metadata_prefill.ENABLED and growth.ENABLED):
        raise RuntimeError("Required migrated DeepSeek optimizations failed to activate")
    memory_policy._PREFILL_HEADROOM_SAFETY["custom"] = .995
    memory_policy._PREFILL_ABORT_MARGIN["custom"] = .995
    original_init = server.init_server
    effective = profile["effective_settings"]

    def initialize(*args, **kwargs):
        config = kwargs.get("scheduler_config") or SchedulerConfig()
        config.prefill_step_size = effective["prefill_step_size"]
        config.prefill_peak_target_gib = effective.get("prefill_peak_target_gib")
        config.max_num_seqs = effective["max_concurrent_requests"]
        config.prefill_speed_priority = False
        config.preserved_prefill_max_defer_seconds = effective["preserved_prefill_max_defer_seconds"]
        config.paged_ssd_cache_dir = None
        config.paged_ssd_cache_max_size = 0
        config.hot_cache_only, config.hot_cache_max_size = False, 0
        kwargs["scheduler_config"] = config
        return original_init(*args, **kwargs)

    server.init_server = initialize
    if cold_prefix_baseline:
        from omlx.patches.deepseek_v41.prefix_cache import ExactPrefixCache
        original_prepare = ExactPrefixCache.prepare
        @functools.wraps(original_prepare)
        def prepare_cold(manager, *args, **kwargs):
            manager.clear()
            return original_prepare(manager, *args, **kwargs)
        ExactPrefixCache.prepare = prepare_cold
    identity = runtime_identity(profile)
    launch_identity = {"source": str(Path(profile["source"]).resolve()),
                       "profile_sha256": sha(PROFILE),
                       "runtime_fingerprint": hashlib.sha256(
                           json.dumps(identity, sort_keys=True).encode()).hexdigest()}
    cache_config = dict(profile["cache"])
    shutdown = Path(profile["state_path"]) / "logs/cache-shutdown.json"
    runtime_type = candidate_cache_runtime(CacheRuntime)
    runtime = runtime_type(server, cache_config, identity, shutdown).install()
    stats = {"metadata_prefill_direct_calls": 0, "metadata_dense_expert_calls": 0,
             "hc_fuse_calls": 0, "batched_verify_calls": 0, "native_single_decode_calls": 0}
    lock = threading.Lock()

    count = functools.partial(count_runtime_call, stats=stats, lock=lock)

    count(native_metadata_prefill, "forward", "metadata_prefill_direct_calls")
    count(native_metadata, "_dense_expert", "metadata_dense_expert_calls")
    from omlx.patches.deepseek_v41 import hc_fuse, batched_verify
    count(hc_fuse, "block_forward", "hc_fuse_calls")
    count(batched_verify, "forward", "batched_verify_calls")
    count(attention_fast, "packed_sparse_attention_single_decode", "native_single_decode_calls")

    @server.app.get("/api/runtime-profile", dependencies=[Depends(server.verify_api_key)])
    def runtime_profile():
        actual = runtime_actual(server, profile["model_id"])
        actual["physical_memory_guard"] = physical_guard
        actual["native_load_memory_guard"] = native_load_guard
        return {"ready": runtime.ready, "error": runtime.error, "release": profile["release"],
                "upstream_commit": profile["upstream_commit"], "model_id": profile["model_id"],
                "pid": os.getpid(), "instance_token": os.environ.get("OMLX_DS_INSTANCE_TOKEN"), "settings": {**effective, **actual}, "actual": actual,
                "launch_identity": launch_identity,
                "cold_prefix_baseline": bool(cold_prefix_baseline)}

    @server.app.get("/api/v41-prefix-cache", dependencies=[Depends(server.verify_api_key)])
    def cache_status():
        return runtime.snapshot()

    @server.app.get("/api/v41-optimizations", dependencies=[Depends(server.verify_api_key)])
    def optimizations():
        pool = server._server_state.engine_pool
        if pool is not None and any(entry.is_loading or entry.in_use or pool._entry_has_scheduler_work(entry)
                                    for entry in pool._entries.values()):
            raise HTTPException(status_code=409, detail="Engine must be idle for weight inventory")
        with lock:
            counters = dict(stats)
        from inspect_omlx_backport_residency import inventory
        return {"profile": profile, "pid": os.getpid(), "stats": counters,
                "prefix_cache": runtime.snapshot(), "actual_growth": {"enabled": growth.ENABLED, "pingpong": growth.PINGPONG},
                "actual": runtime_actual(server, profile["model_id"]),
                "loaded_weight_inventory": inventory(server._server_state),
                "runtime_fingerprint": hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()}

    from request_archive import attach_before_run, load_config
    archive = load_config(Path(profile["state_path"]))
    if archive.pop("enabled"):
        attach_before_run(None, archive)
    print("Migrated oMLX DeepSeek launch: " + json.dumps({"target": profile["target_version"],
          "commit": profile["target_commit"], "host": host, "port": port,
          "model": profile["model_id"], "effective_settings": effective}), flush=True)
    sys.argv = ["omlx", "serve", "--model-dir", str(registry), "--base-path", profile["base_path"],
                "--host", host, "--port", str(port), "--max-concurrent-requests", str(effective["max_concurrent_requests"]),
                "--memory-guard-gb", str(effective["memory_guard_gb"])]
    cli_main()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("serve", "profile"))
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--qualification", action="store_true")
    parser.add_argument("--cold-prefix-baseline", action="store_true")
    args = parser.parse_args()
    profile = read_profile()
    if args.action == "profile":
        print(json.dumps(profile, indent=2))
        return
    serve(profile, args.host or profile.get("host", "127.0.0.1"), args.port or profile.get("port", 8080),
          qualification=args.qualification, cold_prefix_baseline=args.cold_prefix_baseline)


if __name__ == "__main__":
    main()
