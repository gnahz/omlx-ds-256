"""FIFO entrance to the complete inference pipeline, without model imports.

The HTTP owner, preprocessing worker and scheduler hold independent references.
A disconnected client cannot free capacity while its worker or KV cleanup lives.
"""
from __future__ import annotations

import asyncio
from collections import OrderedDict
from dataclasses import dataclass, field
import threading
import time


@dataclass
class _Entry:
    model_id: str
    loop: asyncio.AbstractEventLoop
    ready: asyncio.Event
    started: float
    admitted: bool = False
    owners: set[str] = field(default_factory=lambda: {"http"})


class RequestAdmission:
    def __init__(self):
        self._lock = threading.RLock()
        self._entries: OrderedDict[str, _Entry] = OrderedDict()
        self._limits: dict[str, int] = {}
        self._paused: set[str] = set()

    def _advance(self, model_id):
        if model_id in self._paused:
            return
        active = sum(e.admitted for e in self._entries.values() if e.model_id == model_id)
        for entry in self._entries.values():
            if active >= self._limits[model_id]:
                break
            if entry.model_id == model_id and not entry.admitted:
                entry.admitted = True
                active += 1
                entry.loop.call_soon_threadsafe(entry.ready.set)

    def pause(self, model_id, paused=True):
        with self._lock:
            if paused:
                self._paused.add(model_id)
            else:
                self._paused.discard(model_id)
                if model_id in self._limits:
                    self._advance(model_id)

    def model_ids(self, request_ids):
        with self._lock:
            return {self._entries[rid].model_id for rid in request_ids if rid in self._entries}

    async def acquire(self, request_id, model_id, *, limit=4, disconnected=None):
        if type(limit) is not int or limit <= 0:
            raise ValueError("Admission limit must be a positive integer")
        entry = _Entry(model_id, asyncio.get_running_loop(), asyncio.Event(), time.monotonic())
        with self._lock:
            if request_id in self._entries:
                raise ValueError("Duplicate admission request ID")
            self._entries[request_id] = entry
            self._limits[model_id] = limit
            self._advance(model_id)
        try:
            while True:
                if disconnected is not None and await disconnected():
                    raise asyncio.CancelledError("Client disconnected in entrance queue")
                if entry.ready.is_set():
                    return
                try:
                    await asyncio.wait_for(entry.ready.wait(), timeout=0.25)
                except TimeoutError:
                    pass
        except BaseException:
            self.release(request_id)
            raise

    def retain(self, request_id, owner):
        with self._lock:
            entry = self._entries.get(request_id)
            if entry is not None:
                if not entry.admitted:
                    raise RuntimeError("Queued request cannot start inference work")
                entry.owners.add(owner)

    def release(self, request_id, owner="http"):
        with self._lock:
            entry = self._entries.get(request_id)
            if entry is None:
                return
            entry.owners.discard(owner)
            if not entry.owners:
                del self._entries[request_id]
                self._advance(entry.model_id)

    def snapshot(self, model_id):
        with self._lock:
            now = time.monotonic()
            entries = [(rid, e) for rid, e in self._entries.items() if e.model_id == model_id]
            queued = [{"request_id": rid, "queue_position": i,
                       "elapsed_seconds": max(0.0, now - e.started)}
                      for i, (rid, e) in enumerate(((rid, e) for rid, e in entries if not e.admitted), 1)]
            draining = [{"request_id": rid, "kind": "pipeline_cleanup", "phase": "draining",
                         "elapsed_seconds": max(0.0, now - e.started)}
                        for rid, e in entries if e.admitted and "http" not in e.owners]
            return {"draining": draining, "queued": queued, "queued_requests": len(queued),
                    "admission_paused_for_decode": model_id in self._paused,
                    "admitted_requests": sum(e.admitted for _, e in entries),
                    "request_limit": self._limits.get(model_id, 4)}


_admission = RequestAdmission()


def get_request_admission():
    return _admission
