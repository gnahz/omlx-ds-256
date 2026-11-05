# SPDX-License-Identifier: Apache-2.0
"""Metadata-only visibility for requests before scheduler admission."""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Callable
from concurrent.futures import Executor
from dataclasses import dataclass
from typing import TypeVar

from .request_admission import get_request_admission


_Result = TypeVar("_Result")


@dataclass
class _Preparation:
    model_id: str
    started_at: float
    updated_at: float
    phase: str = "preparing_request"
    prompt_tokens: int | None = None
    worker_pending: bool = False
    transport_closed: bool = False


class RequestPreparationTracker:
    def __init__(self) -> None:
        self._entries: dict[str, _Preparation] = {}
        self._lock = threading.Lock()

    def begin(self, request_id: str, model_id: str) -> None:
        now = time.monotonic()
        with self._lock:
            self._entries[request_id] = _Preparation(model_id, now, now)

    def set_prompt_tokens(self, request_id: str, prompt_tokens: int) -> None:
        with self._lock:
            entry = self._entries.get(request_id)
            if entry is not None and not entry.transport_closed:
                entry.prompt_tokens = prompt_tokens
                entry.updated_at = time.monotonic()

    def close(self, request_id: str) -> None:
        """Remove the request unless its synchronous executor work is draining."""
        with self._lock:
            entry = self._entries.get(request_id)
            if entry is None or entry.transport_closed:
                return
            if entry.worker_pending:
                entry.transport_closed = True
                entry.phase = "cancelling"
                entry.updated_at = time.monotonic()
            else:
                self._entries.pop(request_id)

    def get_model_activities(self, model_id: str) -> list[dict]:
        now = time.monotonic()
        with self._lock:
            return [
                {
                    "request_id": request_id,
                    "kind": "request_preparation",
                    "phase": entry.phase,
                    "elapsed_seconds": max(0.0, now - entry.started_at),
                    "last_activity_age_seconds": max(0.0, now - entry.updated_at),
                    "prompt_tokens": entry.prompt_tokens,
                }
                for request_id, entry in self._entries.items()
                if entry.model_id == model_id
            ]

    async def prepare_prompt(
        self,
        request_id: str | None,
        executor: Executor,
        prepare: Callable[..., _Result],
        *args,
    ) -> _Result:
        return await self._run_on_executor(
            request_id, executor, "waiting_for_executor", "preparing_prompt", prepare, args
        )

    async def submit_to_scheduler(
        self,
        request_id: str,
        executor: Executor,
        add_request: Callable[..., None],
        request,
    ) -> None:
        await self._run_on_executor(
            request_id,
            executor,
            "waiting_for_scheduler",
            "waiting_for_scheduler",
            add_request,
            (request,),
        )

    async def _run_on_executor(
        self,
        request_id: str | None,
        executor: Executor,
        waiting_phase: str,
        running_phase: str,
        operation: Callable[..., _Result],
        args: tuple,
    ) -> _Result:
        with self._lock:
            entry = self._entries.get(request_id) if request_id is not None else None
            if entry is not None:
                entry.worker_pending = True
                entry.phase = waiting_phase
                entry.updated_at = time.monotonic()
        if request_id is None or entry is None:
            return await asyncio.get_running_loop().run_in_executor(
                executor, operation, *args
            )

        # Keep the actual executor future so cancellation can tell whether the
        # queued work was discarded or is still running on the MLX thread.
        get_request_admission().retain(request_id, "worker")
        try:
            work = executor.submit(
                self._run_worker, request_id, running_phase, operation, args
            )
        except BaseException:
            self._worker_finished(request_id)
            raise

        try:
            return await asyncio.wrap_future(work)
        except asyncio.CancelledError:
            self.close(request_id)
            if work.cancel():
                self._worker_finished(request_id)
            raise

    def _run_worker(
        self, request_id: str, phase: str, operation: Callable[..., _Result], args: tuple
    ) -> _Result:
        with self._lock:
            entry = self._entries[request_id]
            if not entry.transport_closed:
                entry.phase = phase
                entry.updated_at = time.monotonic()
        try:
            return operation(*args)
        finally:
            self._worker_finished(request_id)

    def _worker_finished(self, request_id: str) -> None:
        get_request_admission().release(request_id, "worker")
        with self._lock:
            entry = self._entries.get(request_id)
            if entry is None:
                return
            if entry.transport_closed:
                self._entries.pop(request_id)
            else:
                entry.worker_pending = False
                entry.phase = "preparing_request"
                entry.updated_at = time.monotonic()


_tracker = RequestPreparationTracker()


def get_request_preparation_tracker() -> RequestPreparationTracker:
    return _tracker
