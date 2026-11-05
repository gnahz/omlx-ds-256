"""Service-scoped exact-cache lifecycle and native admin SSD-L2 bridge.

All MLX/cache mutations are serialized on the owning engine executor. Restart
snapshots and the request-time SSD L2 remain separate namespaces and controls.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, suppress
import functools
import json
import logging
import os
from pathlib import Path
import time

logger = logging.getLogger(__name__)


class CacheRuntime:
    def __init__(self, server, config, fingerprint, shutdown_receipt):
        self.server = server
        self.config = config
        self.fingerprint = fingerprint
        self.shutdown_receipt = Path(shutdown_receipt)
        self.models = {}
        self.ready = False
        self.error = None
        self.cores = {}
        self._admin_lock = asyncio.Lock()
        self.native_settings_lock = asyncio.Lock()
        self._applied_cache_values = None

    @property
    def l2_requested(self):
        return bool(self.config.get("l2_enabled", False))

    def native_policy(self, settings):
        """Resolve the same native SSD setting shown and edited by the dashboard."""
        cache = settings.cache
        expected = Path(self.config["native_ssd_cache_dir"]).resolve()
        if cache.get_ssd_cache_dir(settings.base_path).resolve() != expected:
            raise ValueError("Exact-cache SSD directory is fixed by the service profile")
        if cache.get_hot_cache_max_size_bytes() != 0:
            raise ValueError(f"Exact-cache RAM is fixed by the {self.config['capacity_gib']:g} GiB service profile; "
                             "native paged hot cache stays disabled")
        if cache.hot_cache_write_through:
            raise ValueError("Native paged hot-cache write-through is unavailable for the exact-cache backend")
        maximum = cache.get_ssd_cache_max_size_bytes(settings.base_path)
        if maximum <= 0:
            raise ValueError("ssd_cache_max_size must be positive")
        return {"enabled": self.l2_requested and cache.enabled and not cache.hot_cache_only,
                "directory": str(expected / "deepseek_v41_exact_l2_v1"), "max_bytes": maximum}

    def validate_native_settings_request(self, request, settings):
        """Reject unsupported cache controls before the shared settings mutate."""
        import copy
        candidate = copy.copy(settings)
        candidate.cache = copy.copy(settings.cache)
        fields = {"cache_enabled": "enabled", "ssd_cache_dir": "ssd_cache_dir",
                  "ssd_cache_max_size": "ssd_cache_max_size", "hot_cache_only": "hot_cache_only",
                  "hot_cache_write_through": "hot_cache_write_through", "hot_cache_max_size": "hot_cache_max_size"}
        for public, private in fields.items():
            value = getattr(request, public, None)
            if value is not None:
                setattr(candidate.cache, private, value)
        self.native_policy(candidate)
        # These are constructed only by the generic paged/GDN backend. Accept
        # the dashboard's unchanged form fields, reject a policy change.
        for field in ("initial_cache_blocks", "gdn_ssd_pending_max_size"):
            value = getattr(request, field, None)
            if value is not None and value != getattr(settings.cache, field, None):
                raise ValueError(f"{field} belongs to the disabled native paged-cache backend")
        split = getattr(request, "gdn_ssd_split_enabled", None)
        if split is not None and split != settings.cache.get_gdn_ssd_split_enabled():
            raise ValueError("GDN sidecars belong to the disabled native paged-cache backend")
        precision = getattr(request, "gdn_sidecar_precision", None)
        if precision is not None and precision != settings.cache.gdn_sidecar_state_dtype:
            raise ValueError("GDN precision belongs to the disabled native paged-cache backend")
        storage = getattr(request, "gdn_snapshot_storage", None)
        if storage is not None and storage != settings.cache.get_gdn_snapshot_storage():
            raise ValueError("GDN snapshot storage belongs to the disabled native paged-cache backend")

    def _configure_l2(self, manager, policy):
        if not self.l2_requested:
            return
        if policy["enabled"]:
            existing = manager.snapshot_stats().get("l2", {})
            if existing.get("enabled"):
                manager.set_l2_max_bytes(policy["max_bytes"], timeout_seconds=self.config["shutdown_timeout_seconds"] - 10)
            else:
                identity = {"deployment": self.fingerprint, "draft_signature": manager.signature}
                manager.configure_l2(policy["directory"], identity, policy["max_bytes"])
        else:
            if not manager.flush_l2(timeout_seconds=self.config["shutdown_timeout_seconds"] - 10):
                raise TimeoutError("Exact SSD L2 could not drain before disabling it")
            manager.close_l2(wait=False)

    async def apply_native_settings(self, settings):
        """Apply SSD enablement/capacity in place, without unloading a model."""
        policy = self.native_policy(settings)
        async with self._admin_lock:
            try:
                for name, core in list(self.cores.items()):
                    def apply(core=core, name=name):
                        manager = self.manager(core)
                        self._configure_l2(manager, policy)
                        self.models[name]["cache"] = manager.snapshot_stats()
                    await self.on_engine(core, apply)
            except Exception:
                if self._applied_cache_values is not None:
                    for key, value in self._applied_cache_values.items():
                        setattr(settings.cache, key, value)
                raise
            self._remember_native_settings(settings)
        return True, "Exact-prefix SSD L2 settings applied without reloading models"

    def _remember_native_settings(self, settings):
        self._applied_cache_values = {key: getattr(settings.cache, key) for key in
            ("enabled", "ssd_cache_dir", "ssd_cache_max_size", "hot_cache_only",
             "hot_cache_max_size", "hot_cache_write_through")}

    async def clear_native_cache(self, *, ssd=False):
        """Clear only the owned requested tier, serialized with inference."""
        cleared = 0
        async with self._admin_lock:
            if not self.cores:
                raise RuntimeError("Load the exact-cache model before clearing its owned cache tiers")
            for name, core in list(self.cores.items()):
                def clear(core=core, name=name):
                    manager = self.manager(core)
                    if ssd:
                        count = manager.clear_l2(timeout_seconds=self.config["shutdown_timeout_seconds"] - 10)
                    else:
                        # Complete pending immutable writes before dropping RAM,
                        # so this operation can demonstrate a genuine L2 hit.
                        if self.l2_requested:
                            if not manager.flush_l2(timeout_seconds=self.config["shutdown_timeout_seconds"] - 10):
                                raise TimeoutError("Exact SSD L2 write did not complete; RAM cache was preserved")
                        count = len(manager.entries)
                        manager.clear()
                    self.models[name]["cache"] = manager.snapshot_stats()
                    return int(count or 0)
                cleared += await self.on_engine(core, clear)
        return {"status": "ok", "backend": "deepseek_v41_exact_l2_v1",
                "total_deleted" if ssd else "total_cleared": cleared,
                "distributed_ranks": 0, **({} if ssd else {"bytes_reclaimed": 0})}

    async def on_engine(self, core, fn):
        import mlx.core as mx
        def invoke():
            with mx.stream(core._mlx_stream):
                return fn()
        return await asyncio.get_running_loop().run_in_executor(core._mlx_executor, invoke)

    def seed_guard(self, scheduler):
        # Loading engines are not yet published in the pool. Seed their guard
        # with the same live enforcer values before restoring any checkpoint.
        enforcer = self.server._server_state.process_memory_enforcer
        if enforcer is None or not enforcer.prefill_memory_guard:
            raise RuntimeError("Exact-cache startup requires the memory guard")
        breakdown = enforcer._get_ceiling_breakdown()
        reserve = enforcer._hot_cache_reserved_bytes()
        ceiling = enforcer._scheduler_limit_bytes(breakdown["hard_limit"], reserved=reserve)
        scheduler._prefill_memory_guard = True
        scheduler._memory_hard_limit_bytes = ceiling
        scheduler._memory_limit_bytes = int(ceiling * enforcer._soft_threshold)
        scheduler._memory_hard_watermark_bytes = int(ceiling * enforcer._hard_threshold)
        scheduler._memory_abort_limit_bytes = enforcer._scheduler_limit_bytes(
            enforcer._get_abort_limit_bytes(), reserved=reserve)
        scheduler._prefill_abort_margin = enforcer._get_prefill_abort_margin()
        scheduler._memory_limits_propagated = True

    async def initialize(self, core):
        name = core.config.model_name
        self.cores[name] = core
        entry = self.models[name] = {"state": "restoring", "cache": {}, "completed": []}
        def restore():
            self.seed_guard(core.scheduler)
            manager = core.scheduler._configure_exact_prefix_cache()
            if manager is None:
                raise RuntimeError("DeepSeek exact-prefix manager was not enabled")
            core._service_exact_cache_manager = manager
            if self.config["persistence_enabled"]:
                identity = {"deployment": self.fingerprint, "draft_signature": manager.signature}
                manager.configure_persistence(self.config["directory"], identity,
                    interval_seconds=self.config["save_interval_seconds"])
                manager.restore_persisted()
            if self.l2_requested:
                self._configure_l2(manager, self.native_policy(self.server._server_state.global_settings))
                self._remember_native_settings(self.server._server_state.global_settings)
            entry["cache"] = manager.snapshot_stats()
        await self.on_engine(core, restore)
        entry["state"] = "ready"
        core._service_exact_cache_task = asyncio.create_task(self.periodic(core))

    def manager(self, core):
        manager = core.scheduler._configure_exact_prefix_cache()
        if manager is not core._service_exact_cache_manager:
            raise RuntimeError("Exact-cache draft configuration changed; restart is required")
        return manager

    async def periodic(self, core):
        name = core.config.model_name
        try:
            while True:
                await asyncio.sleep(1)
                def step():
                    scheduler = core.scheduler
                    manager = self.manager(core)
                    if manager is None:
                        return
                    # Re-check on the executor after any preceding request step.
                    if self.config["persistence_enabled"] and not scheduler.has_requests():
                        manager.persistence_step()
                    if self.l2_requested and not scheduler.has_requests():
                        manager.l2_step()
                    self.models[name]["cache"] = manager.snapshot_stats()
                    running = len(getattr(scheduler, "running", {}))
                    previous = self.models[name].get("scheduler", {})
                    self.models[name]["scheduler"] = {
                        "max_num_seqs": getattr(getattr(scheduler, "config", None), "max_num_seqs", None),
                        "running_requests": running,
                        "waiting_requests": len(getattr(scheduler, "waiting", [])),
                        "prefilling_requests": len(getattr(scheduler, "prefilling", [])),
                        "max_observed_running": max(running, previous.get("max_observed_running", 0)),
                    }
                    self.models[name]["completed"] = list(
                        getattr(scheduler, "exact_prefix_completed", []))
                await self.on_engine(core, step)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            logger.exception("Exact cache maintenance failed")
            self.models[name]["maintenance_error"] = f"{type(error).__name__}: {error}"
            self.models[name]["state"] = "failed"
            self.ready = False
            self.error = f"{type(error).__name__}: {error}"

    async def flush(self, core, deadline=None):
        task = getattr(core, "_service_exact_cache_task", None)
        if task:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        name = core.config.model_name
        if name not in self.models:
            return
        self.ready = False
        self.models[name]["state"] = "saving"
        deadline = deadline or time.monotonic() + self.config["shutdown_timeout_seconds"] - 10
        result = {"pid": os.getpid(), "saved": False, "at": time.time()}
        try:
            while True:
                def step():
                    manager = self.manager(core)
                    pending = manager.persistence_step(force=True) if self.config["persistence_enabled"] else False
                    self.models[name]["cache"] = manager.snapshot_stats()
                    if not pending:
                        persistence = self.models[name]["cache"].get("persistence", {})
                        manager.close_persistence(wait=False)
                        if self.l2_requested:
                            flushed = manager.flush_l2(timeout_seconds=max(0.1, deadline - time.monotonic()))
                            manager.close_l2(wait=False)
                            if not flushed:
                                raise TimeoutError("Exact SSD L2 shutdown write did not complete")
                        if self.config["persistence_enabled"] and (
                                not persistence.get("enabled") or persistence.get("dirty") or persistence.get("last_error")):
                            raise RuntimeError("Checkpoint save incomplete: " + str(persistence.get("last_error")))
                        manager.clear()
                    return pending
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Exact cache shutdown save exceeded its deadline")
                pending = await asyncio.wait_for(self.on_engine(core, step), timeout=remaining)
                if not pending:
                    break
                await asyncio.sleep(.05)
            result.update(saved=True, cache=self.models[name]["cache"])
        except Exception as error:
            result["error"] = f"{type(error).__name__}: {error}"
            logger.exception("Exact cache shutdown save failed")
        core._service_exact_cache_manager = None
        self.cores.pop(name, None)
        self.models[name]["state"] = "stopped"
        self.shutdown_receipt.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.shutdown_receipt.with_suffix(".tmp")
        temporary.write_text(json.dumps(result, indent=2) + "\n")
        temporary.replace(self.shutdown_receipt)

    def snapshot(self):
        return {"ready": self.ready, "pid": os.getpid(), "error": self.error,
                "read_policy": "on_demand" if self.l2_requested else "startup_only",
                "backend": "deepseek_v41_exact_l2_v1" if self.l2_requested else "exact_restart_snapshots",
                "models": self.models}

    def install(self):
        from omlx.engine_core import EngineCore
        from starlette.responses import JSONResponse
        original_start, original_stop = EngineCore.start, EngineCore.stop
        runtime = self
        self.server._server_state.exact_prefix_cache_runtime = self

        @functools.wraps(original_start)
        async def start(core):
            if not core.is_running() and getattr(core.model, "model_type", None) == "deepseek_v41":
                if runtime.config["enabled"]:
                    await runtime.initialize(core)
            return await original_start(core)

        @functools.wraps(original_stop)
        async def stop(core):
            deadline = time.monotonic() + runtime.config["shutdown_timeout_seconds"] - 10
            task = getattr(core, "_service_exact_cache_task", None)
            if task:
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
            await original_stop(core)
            if getattr(core, "_service_exact_cache_task", None) is not None:
                await runtime.flush(core, deadline)
                core._service_exact_cache_task = None

        EngineCore.start, EngineCore.stop = start, stop
        original_lifespan = self.server.app.router.lifespan_context

        @asynccontextmanager
        async def lifespan(app):
            async with original_lifespan(app):
                async def preload():
                    try:
                        while not runtime.server._server_state.pinned_preload_complete:
                            await asyncio.sleep(.1)
                        await runtime.server._server_state.engine_pool.get_engine("deepseek-v41-lsq")
                        l2_required = (runtime.l2_requested and runtime.native_policy(
                            runtime.server._server_state.global_settings)["enabled"])
                        if runtime.config["enabled"] and not any(
                                entry["state"] == "ready" and entry["cache"].get("enabled") and
                                (not runtime.config["persistence_enabled"] or
                                 entry["cache"].get("persistence", {}).get("enabled")) and
                                (not l2_required or entry["cache"].get("l2", {}).get("enabled"))
                                for entry in runtime.models.values()):
                            raise RuntimeError("Model loaded without the required exact-cache runtime")
                        runtime.ready = True
                    except Exception as error:
                        runtime.error = f"{type(error).__name__}: {error}"
                        logger.exception("Exact cache service initialization failed")
                task = asyncio.create_task(preload())
                try:
                    yield
                finally:
                    runtime.ready = False
                    if not task.done():
                        task.cancel()
                    with suppress(asyncio.CancelledError):
                        await task
        self.server.app.router.lifespan_context = lifespan

        @self.server.app.middleware("http")
        async def readiness(request, call_next):
            if not runtime.ready and (request.url.path == "/health" or request.url.path in (
                    "/v1/chat/completions", "/v1/completions", "/v1/responses", "/v1/messages")):
                return JSONResponse({"error": {"message": "Model and prefix cache are loading",
                    "type": "service_unavailable"}}, status_code=503)
            return await call_next(request)
        return self
