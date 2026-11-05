"""Incremental pipeline reservations on top of the unmodified physical footprint.

Use current progress and actual allocation boundaries; never output limits. Credit only a confirmed materialized KV
copy; charge the scheduler's allocator-reclaim allowance once across requests.
No credit is taken for a cache hit, a pending free, or an unrelated RSS change.
"""
from dataclasses import dataclass, asdict
from .kv_capacity import growth_capacity
from collections import deque
import copy
import math

class TransientMemoryPressureError(RuntimeError):
    """Retryable resource pressure, distinct from invalid input or cancellation."""


GIB = 1024**3
PROCESS_CEILING = 240 * GIB
SAFETY_RESERVE = 2 * GIB
# This runtime spends measured free RAM without an additional fixed holdback.
# The process ceiling and admission safety reserve remain independent limits.
SYSTEM_FREE_RESERVE = 0


def physical_headroom_budget(footprint, stats):
    """Spend measured free RAM without an additional fixed free-RAM reserve.

    File-backed and speculative pages are not released RAM. In the 256 GiB
    live overlap test macOS compressed other processes while those pages were
    still cached, below the process cap. Do not assume their timely eviction.
    This supplements the existing process/admission ceilings; it never raises
    them or credits a pending allocator release. Missing telemetry stops new
    allocations. The returned reserve field remains for API compatibility.
    """
    valid = isinstance(stats, dict) and type(stats.get("free")) is int and footprint > 0
    free = max(0, stats["free"] - max(0, int(stats.get("speculative", 0)))) if valid else 0
    return {"system_telemetry_available": valid,
            "system_free_bytes": free,
            "system_free_reserve_bytes": SYSTEM_FREE_RESERVE,
            "system_limit_bytes": max(1, int(footprint) + free - SYSTEM_FREE_RESERVE) if valid else 1}


@dataclass(frozen=True)
class RequestMemoryEnvelope:
    context_tokens: int
    resident_bytes: int
    prefill_bytes: int
    decode_bytes: int
    total_bytes: int
    credit_cap_bytes: int | None = None
    current_capacity_bytes: int = 0
    next_capacity_bytes: int = 0
    allocation_growth_bytes: int = 0
    step_tokens: int = 5
    prefill_chunk_tokens: int = 0


def estimate_envelope(monitor, prompt_tokens, max_tokens, step_size, predict, *,
                      decode_predict=None, step_tokens=5, caches=None, decoding=False):
    # max_tokens is retained in this internal signature for compatibility only.
    context = max(1, int(prompt_tokens))
    span = max(1, int(step_tokens))
    query = max(1, min(int(step_size), max(1, context - 1)))
    profile = getattr(monitor, "_prefill_memory_profile", None)
    slots = getattr(profile, "allocation_slots", ())
    slots = slots if isinstance(slots, tuple) else ()
    next_capacity = sum(growth_capacity((context + span + ratio - 1) // ratio,
                                        (span + ratio - 1) // ratio) * width
                        for ratio, width in slots)
    if not slots:
        next_capacity = int(monitor.estimate_resident_kv_bytes(
            growth_capacity(context + span, span), chunk_tokens=span))
    # Without a registered cache, allow current storage, the second bank and
    # COW at the next allocation boundary. No hypothetical future output.
    resident = 3 * next_capacity
    cap = next_capacity
    current_capacity = 0
    allocation_growth = next_capacity
    layout = allocation_layout(caches, context, span) if decoding else None
    if layout is not None:
        current_capacity, next_capacity, allocation_growth = layout
        resident = current_capacity + allocation_growth
        cap = current_capacity
    prefill = 0 if decoding else math.ceil(predict(query, max(0, context - query)))
    verify = math.ceil((decode_predict or predict)(span, context))
    if resident < 0 or prefill < 0 or verify < 0 or resident + prefill + verify == 0:
        raise ValueError("Request memory estimate is unavailable")
    return RequestMemoryEnvelope(context, resident, prefill, resident + verify,
                                 resident + max(prefill, verify), cap,
                                 current_capacity, next_capacity, allocation_growth, span,
                                 0 if decoding else query)


def allocation_layout(caches, context, span):
    """Read host geometry only. Bound one new/COW bank per growing slot.

    The old active and idle banks remain charged until observed release. A
    retained view may force COW even when the idle bank has enough capacity.
    No array contents, materialization or GPU synchronization are requested.
    """
    if not isinstance(caches, (list, tuple)) or not caches:
        return None
    current, future, extra, seen = 0, 0, 0, set()
    for cache in caches:
        if type(cache).__name__ != "DeepseekV41Cache":
            return None
        state = vars(cache)
        dense, registry = state.get("_dense"), state.get("_ds41_buffers", {})
        if state.get("_rows") is not None or not isinstance(dense, (list, tuple)):
            return None
        ratio = max(1, int(state.get("compress_ratio", 1)))
        for slot in (2, 3):
            value = dense[slot]
            if value is None:
                continue
            shape = getattr(value, "shape", ())
            # Non-source layers carry empty packed placeholders. They never
            # append to these slots and have no growth registry by design.
            if state.get("compress_ratio") == 0 and len(shape) > 1 and shape[1] == 0:
                continue
            entry = registry.get(slot, {})
            chain = value is entry.get("view")
            buffer, idle = entry.get("buffer"), entry.get("idle")
            if buffer is None or len(buffer.shape) < 2 or buffer.shape[1] <= 0:
                return None
            width = int(buffer.nbytes) // int(buffer.shape[1])
            rows = (span + ratio - 1) // ratio
            end = max(int(entry["end"]) + rows, (context + span + ratio - 1) // ratio)
            capacity = (int(idle.shape[1]) if chain and idle is not None and idle.shape[1] >= end
                        and rows <= 64 and idle.shape[0] == buffer.shape[0]
                        and idle.shape[2:] == buffer.shape[2:]
                        and getattr(idle, "dtype", None) == getattr(buffer, "dtype", None)
                        else growth_capacity(end, rows))
            for bank in (buffer, idle):
                if bank is not None and id(bank) not in seen:
                    seen.add(id(bank)); current += int(bank.nbytes)
            future += capacity * width
            # Include append repair scratch as well as the possible full COW.
            repair = max(rows, end - min(int(entry.get("valid", entry["end"])), int(entry["end"]))) if chain else rows
            extra += capacity * width + 2 * repair * width
    return (current, future, extra) if seen else None


def materialized_kv_buffers(caches, *, include_idle=False):
    """Metadata only, after the owner's existing completed-prefill eval fence.

    Read registered growing backing arrays, never .state/.cache properties,
    views or arbitrary model tensors. Unsupported layouts get no credit.
    """
    pending = list(caches or ())
    seen, result = set(), {}
    while pending:
        cache = pending.pop()
        if id(cache) in seen or type(cache).__name__ != "DeepseekV41Cache":
            continue
        seen.add(id(cache))
        state = vars(cache)
        rows = state.get("_rows")
        if isinstance(rows, (tuple, list)):
            pending.extend(rows)
            continue
        registry = state.get("_ds41_buffers")
        dense = state.get("_dense")
        if not isinstance(registry, dict) or not isinstance(dense, (tuple, list)) or len(dense) != 7:
            continue
        for slot in (2, 3):
            entry = registry.get(slot)
            if not isinstance(entry, dict) or (not include_idle and dense[slot] is not entry.get("view")):
                continue
            # The default prefill path credits only its matching active bank.
            # After a completed decode fence, retained registered banks remain
            # resident even when rollback replaced the visible view.
            buffer = entry.get("buffer")
            size = getattr(buffer, "nbytes", None)
            if type(size) is int and size > 0:
                result[id(buffer)] = size
            if include_idle:
                idle = entry.get("idle")
                size = getattr(idle, "nbytes", None)
                if type(size) is int and size > 0:
                    result[id(idle)] = size
    return result


class RequestMemoryBudget:
    def __init__(self):
        self.reservations: dict[str, RequestMemoryEnvelope] = {}
        self.last_decision = {}
        self.last_resize_decision = {}
        self.decoding: set[str] = set()
        self.shared_reclaim_bytes = 0
        # Only enabled for a model with an independently bounded decode
        # workspace. Other models retain the existing conservative charge.
        self.separate_decode_workspace = False
        self._materialized: dict[str, dict[int, int]] = {}

    @staticmethod
    def limit(dynamic_limit):
        return min(int(dynamic_limit), PROCESS_CEILING - SAFETY_RESERVE) if dynamic_limit > 0 else 0

    def note_materialized(self, request_id, buffers):
        if request_id in self.reservations:
            self._materialized[request_id] = dict(buffers)

    def _credits(self, reservations=None):
        seen, credits = set(), {}
        reservations = self.reservations if reservations is None else reservations
        for rid, envelope in reservations.items():
            credited = 0
            for identity, size in self._materialized.get(rid, {}).items():
                if identity not in seen:
                    seen.add(identity)
                    credited += size
            # Cap confirmed live residency by this allocation plan. Legacy
            # envelopes retain their original one-copy credit limit.
            credits[rid] = min(envelope.resident_bytes // 3 if envelope.credit_cap_bytes is None else envelope.credit_cap_bytes, credited)
        return credits

    def _individual(self, rid, credits, *, decode=False):
        envelope = self.reservations[rid]
        total = envelope.decode_bytes if decode or rid in self.decoding else envelope.total_bytes
        return max(0, total - credits.get(rid, 0))

    def reserve(self, request_id, envelope, current, dynamic_limit):
        if request_id in self.reservations:
            return True
        held = self.reserved_bytes()
        required = envelope.total_bytes + self._new_prefill_shared_bytes()
        limit = self.limit(dynamic_limit)
        peak = int(current) + held + required
        allowed = limit > 0 and peak <= limit
        self.last_decision = {"request_id": request_id, "allowed": allowed,
                              "current_bytes": int(current), "reserved_bytes": held,
                              "required_bytes": required,
                              "predicted_peak_bytes": peak, "limit_bytes": limit,
                              "envelope": asdict(envelope)}
        if allowed:
            self.reservations[request_id] = envelope
        return allowed

    def try_resize(self, request_id, envelope, current, dynamic_limit, *,
                   headroom_margin=0):
        """Replace a live prefill plan only if the complete new ledger fits.

        Called synchronously by the scheduler at a completed chunk boundary.
        Price a private candidate ledger before changing the live reservation;
        release/reserve would discard confirmed KV credits and create a gap in
        ownership. The candidate keeps insertion order so shared buffers are
        credited once, using each candidate envelope's credit cap.

        ``headroom_margin`` is hysteresis for an optional upgrade, not another
        allocation or a permanent reservation. It cannot increase any ceiling.
        Admission diagnostics remain separate from this optional resize probe.
        """
        previous = self.reservations.get(request_id)
        if previous is None or request_id in self.decoding:
            self.last_resize_decision = {
                "request_id": request_id, "allowed": False,
                "reason": "missing_reservation" if previous is None else "already_decoding",
            }
            return False

        candidate = dict(self.reservations)
        candidate[request_id] = envelope
        credits = self._credits(candidate)
        reserved = sum(max(0, (e.decode_bytes if rid in self.decoding else e.total_bytes)
                           - credits.get(rid, 0))
                       for rid, e in candidate.items()) + self._held_shared_bytes()
        limit = self.limit(dynamic_limit)
        margin = max(0, int(headroom_margin))
        peak = int(current) + reserved
        allowed = limit > 0 and peak + margin <= limit
        self.last_resize_decision = {
            "request_id": request_id, "allowed": allowed,
            "current_bytes": int(current),
            "previous_reserved_bytes": self.reserved_bytes(),
            "reserved_bytes": reserved,
            "predicted_peak_bytes": peak,
            "headroom_margin_bytes": margin,
            "limit_bytes": limit,
            "previous_chunk_tokens": previous.prefill_chunk_tokens,
            "selected_chunk_tokens": envelope.prefill_chunk_tokens,
            "envelope": asdict(envelope),
        }
        if allowed:
            self.reservations[request_id] = envelope
        return allowed

    def refresh(self, request_id, envelope, live_buffers):
        self.reservations[request_id] = envelope
        # Never credit a bank that was replaced since the last completed fence.
        previous = self._materialized.get(request_id, {})
        self._materialized[request_id] = {key: size for key, size in previous.items()
                                        if live_buffers.get(key) == size}

    def mark_decoding(self, request_id):
        if request_id in self.reservations:
            self.decoding.add(request_id)

    def reserved_bytes(self):
        credits = self._credits()
        return (sum(self._individual(rid, credits) for rid in self.reservations)
                + self._held_shared_bytes())

    def _held_shared_bytes(self):
        needs_prefill = any(rid not in self.decoding for rid in self.reservations)
        return self.shared_reclaim_bytes if self.reservations and (
            not self.separate_decode_workspace or needs_prefill) else 0

    def _new_prefill_shared_bytes(self):
        return max(0, self.shared_reclaim_bytes - self._held_shared_bytes())

    def release(self, request_id):
        self.reservations.pop(request_id, None)
        self.decoding.discard(request_id)
        self._materialized.pop(request_id, None)

    def snapshot(self, current, dynamic_limit, blocked_request_id=None):
        limit = self.limit(dynamic_limit)
        reserved = self.reserved_bytes()
        credits = self._credits()
        pending = None
        if blocked_request_id is not None and blocked_request_id == self.last_decision.get("request_id"):
            pending = dict(self.last_decision)
            required = pending["envelope"]["total_bytes"] + self._new_prefill_shared_bytes()
            pending.update(current_bytes=int(current), reserved_bytes=reserved,
                           required_bytes=required,
                           predicted_peak_bytes=int(current) + reserved + required,
                           limit_bytes=limit,
                           shortfall_bytes=max(0, int(current) + reserved + required - limit))
        return {
            "accounting_version": 4,
            "current_bytes": int(current), "reserved_bytes": reserved,
            "limit_bytes": limit, "process_limit_bytes": PROCESS_CEILING,
            "available_bytes": max(0, limit - int(current) - reserved),
            "shared_transient_bytes": self._held_shared_bytes(),
            "prefill_shared_transient_bytes": self.shared_reclaim_bytes,
            "decode_shared_transient_bytes": 0 if self.separate_decode_workspace else self._held_shared_bytes(),
            "decode_workspace_model": "bounded_deepseek" if self.separate_decode_workspace else "legacy_conservative",
            "materialized_kv_credit_bytes": sum(credits.values()),
            "pending": pending,
            "reservations": [
                {"request_id": rid,
                 "phase": "decode" if rid in self.decoding else "prefill",
                 "reserved_bytes": self._individual(rid, credits),
                 "materialized_kv_credit_bytes": credits.get(rid, 0),
                 "remaining_kv_bytes": max(0, (e.credit_cap_bytes if e.credit_cap_bytes is not None else e.resident_bytes // 3) - credits.get(rid, 0)),
                 "copy_reserve_bytes": e.resident_bytes - (e.credit_cap_bytes if e.credit_cap_bytes is not None else e.resident_bytes // 3),
                 "current_capacity_bytes": e.current_capacity_bytes,
                 "next_capacity_bytes": e.next_capacity_bytes,
                 "allocation_growth_bytes": e.allocation_growth_bytes,
                 "step_tokens": e.step_tokens,
                 "prefill_chunk_tokens": e.prefill_chunk_tokens,
                 "decode_workspace_bytes": e.decode_bytes - e.resident_bytes,
                 "context_tokens": e.context_tokens}
                for rid, e in self.reservations.items()
            ],
        }

    def decode_additional_bytes(self, request_ids):
        request_ids = list(request_ids)
        credits = self._credits()
        return (sum(self._individual(rid, credits, decode=True) for rid in request_ids)
                + (self.shared_reclaim_bytes if request_ids and not self.separate_decode_workspace else 0))


class DecodeBudgetRecovery:
    """Bounded recovery; one victim per release window, never a batch abort."""
    INTERVAL = 0.25
    RELEASE_WAIT = 5.0

    def __init__(self):
        self.state = "running"
        self.retry_at = 0.0
        self.deadline = 0.0
        self.episode = 0
        self.sequence = 0
        self.history = deque(maxlen=64)
        self.last_failure = None

    @property
    def blocked(self):
        return self.state != "running"

    def due(self, now):
        return not self.blocked or now >= self.retry_at

    def decide(self, fits, now):
        if fits:
            recovered = self.blocked
            self.state = "running"
            self.retry_at = self.deadline = 0.0
            return "recovered" if recovered else "run"
        self.retry_at = now + self.INTERVAL
        if not self.blocked:
            self.episode += 1
            self.state = "waiting_release"
            self.deadline = now + self.RELEASE_WAIT
            return "reclaim"
        if now >= self.deadline:
            self.state = "releasing_latest"
            self.deadline = now + self.RELEASE_WAIT
            return "release_latest"
        self.state = "waiting_release"
        return "wait"

    def record(self, stage, snapshot, now):
        self.sequence += 1
        event = copy.deepcopy({**snapshot, "event_id": self.sequence,
                               "episode_id": self.episode, "stage": stage,
                               "monotonic_seconds": now})
        self.history.append(event)
        if stage != "recovered":
            self.last_failure = event
        return event

    def snapshot(self):
        return {"state": self.state, "episode_id": self.episode,
                "last_failure": self.last_failure, "events": list(self.history)}
