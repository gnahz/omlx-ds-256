# SPDX-License-Identifier: MIT
"""Exact, bounded, process-local checkpoints of the CED prefill program.

Only complete 8192-token chunks are reusable. Decode states and partial chunk
states are never admitted: their arithmetic and decoder replay differ from a
cold prefill of a longer prompt. Each checkpoint includes the decoder window,
compressed KV, index keys, compressor tails, Engram history and DSpark prime
ring. Model identity is implicit in ownership by one loaded language model.
Image checkpoints additionally prove every effective content/geometry identity;
token IDs alone cannot identify the image embeddings used during prefill.

The engine thread owns all MLX operations. Optional persistence only stages one
checkpoint at a time; a worker writes already materialized CPU bytes.
"""

from collections import OrderedDict
from dataclasses import dataclass
import copy
import hashlib
import json
import logging
import os
import struct
import time
import threading

import mlx.core as mx

from .cache import DeepseekV41Cache
from ..mlx_lm_mtp.deepseek_v4_dspark import DSparkContextCache, _DSparkPrimeContext

logger = logging.getLogger(__name__)
CHUNK = 8192
IMAGE_TOKEN_ID = 129264
FIBONACCI_TAIL = "fibonacci_tail"
FIBONACCI_RAM_MAX_TOKENS = 21 * CHUNK
FIBONACCI_TAIL_MAX_BYTES = 10 * 2**30
_IDENTITY_DOMAIN = b"\0ds41-image-prefix-identity-v1\0"


def _is_fibonacci_checkpoint(end):
    blocks, remainder = divmod(end, CHUNK)
    if blocks < 1 or remainder:
        return False
    current, following = 1, 2
    while current < blocks:
        current, following = following, current + following
    return current == blocks


def _identity_keys(value):
    if (not isinstance(value, (tuple, list)) or not 1 <= len(value) <= 128
            or any(not isinstance(key, str) or len(key) != 64
                   or any(c not in "0123456789abcdef" for c in key) for key in value)):
        raise ValueError("invalid image identity keys")
    return tuple(value)


def prefix_identity(value, end):
    """Validate the complete, ordered image proof stored with one checkpoint."""
    if not isinstance(value, (tuple, list)) or len(value) > 4096:
        raise ValueError("invalid image identity ranges")
    result, previous = [], -1
    for item in value:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise ValueError("invalid image identity range")
        start, keys = item
        if type(start) is not int or not previous < start < end or start < 0:
            raise ValueError("invalid image identity boundary")
        result.append((start, _identity_keys(keys)))
        previous = start
    return tuple(result)


def identity_bytes(identity):
    return (json.dumps(identity, separators=(",", ":")).encode() if identity else b"")


def checkpoint_key(tokens, identity=()):
    digest = hashlib.sha256(b"ds41-ced-8192-v1" + tokens)
    proof = identity_bytes(identity)
    if proof:
        digest.update(_IDENTITY_DOMAIN)
        digest.update(struct.pack("<Q", len(proof)))
        digest.update(proof)
    return digest.digest()


def _request_identity(tokens, image_token_id, extra_keys, extra_key_token_start, extra_key_ranges):
    """Fail closed for missing, contradictory or malformed image provenance.

    Trusted callers provide cumulative identities at actual token boundaries.
    A later identity never influences a checkpoint ending before its boundary.
    """
    first_image = next((i for i, token in enumerate(tokens) if token == image_token_id), None)
    supplied = any(value is not None for value in
                   (extra_keys, extra_key_token_start, extra_key_ranges))
    if not supplied:
        if first_image is not None:
            raise ValueError("image tokens require content identities")
        return ()
    keys = None if extra_keys is None else _identity_keys(extra_keys)
    start = extra_key_token_start
    if start is not None and (type(start) is not int or not 0 <= start < len(tokens)):
        raise ValueError("invalid first image identity boundary")
    if extra_key_ranges is not None:
        ranges = prefix_identity(extra_key_ranges, len(tokens))
        if (not ranges or (keys is not None and ranges[-1][1] != keys)
                or (start is not None and ranges[0][0] != start)):
            raise ValueError("contradictory image identity metadata")
    elif keys is not None:
        ranges = ((0 if start is None else start, keys),)
    else:
        raise ValueError("image identities are missing")
    if first_image is not None and ranges[0][0] > first_image:
        raise ValueError("image identity begins after image tokens")
    return ranges


_copy_kernel = None


def compact_copy(value):
    """Own exactly the live bytes; cache views can retain much larger buffers."""
    global _copy_kernel
    if value is None:
        return None
    if not value.size:
        # An empty compressor tail may still be a view of a whole chunk.
        return mx.zeros(value.shape, value.dtype)
    if mx.default_device() == mx.cpu:
        import numpy as np

        # mx.stream also sets the active default device, so GPU stream
        # contexts retain the native path below. Copy bytes instead of float
        # values to preserve NaN payloads, signed zero and strided-view bits.
        raw = np.asarray(mx.contiguous(value).view(mx.uint8)).copy()
        return mx.array(raw).view(value.dtype).reshape(value.shape)
    if _copy_kernel is None:
        _copy_kernel = mx.fast.metal_kernel(
            name="ds41_prefix_snapshot_bytes",
            input_names=["src", "count"],
            output_names=["dst"],
            source="uint i = thread_position_in_grid.x; if (i < count[0]) dst[i] = src[i];",
        )
    raw = mx.contiguous(value).view(mx.uint8)
    copied = _copy_kernel(
        inputs=[raw, mx.array([raw.size], mx.uint32)],
        grid=(raw.size, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[raw.shape],
        output_dtypes=[mx.uint8],
    )[0]
    return copied.view(value.dtype).reshape(value.shape)


def clone_cache(cache, *, compact=False):
    """New Python/MLX handles; subsequent writes use MLX copy-on-write.

    Do not retain growth buffers, speculative rollback stashes or request
    metadata. These are mutable accelerators, not the committed model state.
    """
    copy = compact_copy if compact else lambda x: None if x is None else mx.array(x)
    result = []
    for item in cache:
        new = DeepseekV41Cache(item.compress_ratio)
        new.cache = [copy(x) for x in item.cache]
        new.left_padding = copy(item.left_padding)
        new.lengths = copy(item.lengths)
        result.append(new)
    ctx = getattr(cache[0], "_omlx_mtp_prime_ctx", None)
    if ctx is not None:
        stages = []
        for stage in ctx.caches:
            copied = DSparkContextCache(stage.max_size)
            copied.offset = stage.offset
            copied.keys = copy(stage.keys)
            stages.append(copied)
        result[0]._omlx_mtp_prime_ctx = _DSparkPrimeContext(
            stages, ctx.expected_target_offset
        )
    return result


def cache_arrays(cache):
    arrays = [x for c in cache for x in c.cache if x is not None]
    arrays += [x for c in cache for x in (c.left_padding, c.lengths) if x is not None]
    ctx = getattr(cache[0], "_omlx_mtp_prime_ctx", None)
    if ctx is not None:
        arrays += [c.keys for c in ctx.caches if c.keys is not None]
    return arrays


@dataclass
class Plan:
    tokens: bytes
    keys: dict
    next_position: int
    valid: bool = True
    identities: dict = None


@dataclass
class Entry:
    tokens: bytes
    cache: list
    size: int
    identity: tuple = ()


class ExactPrefixCache:
    def __init__(self, max_bytes, *, image_token_id=IMAGE_TOKEN_ID, policy="dense"):
        if policy not in ("dense", FIBONACCI_TAIL):
            raise ValueError(f"Unknown DeepSeek exact-prefix checkpoint policy: {policy}")
        self.max_bytes = max(0, int(max_bytes))
        self.image_token_id = image_token_id
        self.policy = policy
        self.ram_checkpoint_max_tokens = None
        self.tail_max_bytes = None
        if policy == FIBONACCI_TAIL:
            self.ram_checkpoint_max_tokens = FIBONACCI_RAM_MAX_TOKENS
            self.tail_max_bytes = FIBONACCI_TAIL_MAX_BYTES
        self.entries = OrderedDict()
        self.bytes = 0
        self.hits = self.stores = self.evictions = 0
        self.ram_hits = self.l2_hits = 0
        self.reused_tokens = self.skipped_memory = self.restore_failures = 0
        self.record_skipped_partial_chunks = self.record_skipped_position = 0
        self.record_skipped_key = self.record_skipped_oversized = 0
        self.record_skipped_invalid_plan = 0
        self.record_skipped_policy = 0
        self.ssd_only_store_attempts = self.ssd_only_store_skipped = 0
        self.ensure_headroom = None
        self._protected_keys = set()
        self._revision = 0
        self._persistence = None
        self._l2 = None
        self._owner_thread = threading.get_ident()
        self._published_stats = {}
        self.snapshot_stats()

    def is_tail_checkpoint(self, end):
        return self.policy == FIBONACCI_TAIL and not _is_fibonacci_checkpoint(end)

    def ram_admits_checkpoint(self, end):
        return self.ram_checkpoint_max_tokens is None or (
            end <= self.ram_checkpoint_max_tokens and not self.is_tail_checkpoint(end)
        )

    def clear(self):
        self.entries.clear()
        self.bytes = 0
        self._revision += 1

    def evict_bytes(self, required):
        """Release least recently used snapshots, except a pinned restore."""
        before = self.bytes
        freed = 0
        for key in list(self.entries):
            if freed >= required:
                break
            if key in self._protected_keys:
                continue
            old = self.entries.get(key)
            if old is None:
                continue
            # Prewrite normally made this cheap. Under pressure a spill may
            # only stage if the existing headroom policy admits its CPU copy;
            # bounded memory takes precedence over retaining a cold entry.
            if self._l2 is not None and not self._l2._staging:
                self._l2.store(key, old)
            old = self.entries.pop(key, None)
            if old is None:
                freed = before - self.bytes
                continue
            self.bytes -= old.size
            freed = before - self.bytes
            self.evictions += 1
            self._revision += 1
        return freed

    def _headroom(self, additional_bytes):
        if self.ensure_headroom is not None and not self.ensure_headroom(additional_bytes):
            self.skipped_memory += 1
            return False
        return True

    def snapshot_stats(self):
        # Dashboard callers run on the asyncio thread. Only the owning engine
        # may inspect live RAM entries or retire an SSD worker. Published
        # dictionaries are never mutated, and every reader gets its own copy.
        if threading.get_ident() != self._owner_thread:
            return copy.deepcopy(self._published_stats)
        result = dict(
            enabled=bool(self.max_bytes), max_bytes=self.max_bytes,
            bytes=self.bytes, checkpoints=len(self.entries), hits=self.hits,
            reused_tokens=self.reused_tokens, stores=self.stores,
            ram_hits=self.ram_hits, l2_hits=self.l2_hits,
            evictions=self.evictions, skipped_memory=self.skipped_memory,
            restore_failures=self.restore_failures,
            record_skipped_partial_chunks=self.record_skipped_partial_chunks,
            record_skipped_position=self.record_skipped_position,
            record_skipped_key=self.record_skipped_key,
            record_skipped_oversized=self.record_skipped_oversized,
            record_skipped_invalid_plan=self.record_skipped_invalid_plan,
            checkpoint_policy=self.policy,
            ram_checkpoint_max_tokens=self.ram_checkpoint_max_tokens,
            record_skipped_policy=self.record_skipped_policy,
            ssd_only_store_attempts=self.ssd_only_store_attempts,
            ssd_only_store_skipped=self.ssd_only_store_skipped,
        )
        result["persistence"] = (
            self._persistence.stats() if self._persistence else {
                "enabled": False, "disk_reads": 0, "restored_checkpoints": 0,
            }
        )
        result["l2"] = self._l2.stats() if self._l2 is not None else {
            "backend": "deepseek_v41_exact_l2_v1", "enabled": False,
            "max_bytes": 0, "disk_bytes": 0, "entries": 0,
            "hits": 0, "misses": 0, "reads": 0, "writes": 0,
            "read_bytes": 0, "write_bytes": 0, "evictions": 0,
            "read_failures": 0, "write_failures": 0,
            "pending_bytes": 0, "pending": False, "last_error": None,
        }
        self._published_stats = copy.deepcopy(result)
        return result

    def configure_l2(self, directory, identity, max_bytes):
        from .prefix_cache_l2 import ExactPrefixL2
        if self._l2 is not None:
            self._l2.close()
        self._l2 = ExactPrefixL2(self, directory, identity, max_bytes)

    def l2_step(self):
        return self._l2.step() if self._l2 is not None else False

    def flush_l2(self, timeout_seconds=None):
        return self._l2.flush(timeout_seconds) if self._l2 is not None else True

    def close_l2(self, wait=True, timeout_seconds=None):
        return self._l2.close(wait, timeout_seconds) if self._l2 is not None else True

    def set_l2_max_bytes(self, max_bytes, timeout_seconds=30):
        if self._l2 is not None:
            self._l2.set_max_bytes(max_bytes, timeout_seconds)

    def clear_l2(self, timeout_seconds=30):
        return self._l2.clear(timeout_seconds) if self._l2 is not None else 0

    def configure_persistence(self, directory, fingerprint, interval_seconds=300):
        from .prefix_cache_store import PrefixCacheStore

        if self._persistence is not None:
            self.close_persistence()
        self._persistence = PrefixCacheStore(self, directory, fingerprint, interval_seconds)

    def restore_persisted(self):
        """Restore the legacy startup-only persistence tier."""
        return self._persistence.restore() if self._persistence else 0

    def persistence_step(self, force=False):
        """Called while idle on the engine thread; stage at most one snapshot."""
        return self._persistence.step(force=force) if self._persistence else False

    @property
    def persistence_pending(self):
        return bool(self._persistence and self._persistence.pending)

    def close_persistence(self, wait=True):
        if self._persistence:
            self._persistence.close(wait=wait)

    def _lookup_plan(self, tokens, step=CHUNK, *, extra_keys=None,
                     extra_key_token_start=None, extra_key_ranges=None):
        """Build the same token/image lookup keys without touching cache state."""
        # Leave the final prompt token for the ordinary generation kickoff.
        if not self.max_bytes or step != CHUNK or len(tokens) <= CHUNK:
            return None
        try:
            identity = _request_identity(tokens, self.image_token_id, extra_keys,
                                         extra_key_token_start, extra_key_ranges)
        except (TypeError, ValueError):
            return None
        packed = struct.pack("<%dI" % len(tokens), *tokens)
        packed_view = memoryview(packed)
        digest = hashlib.sha256(b"ds41-ced-8192-v1")
        keys, identities = {}, {}
        for end in range(CHUNK, len(tokens), CHUNK):
            digest.update(packed_view[(end - CHUNK) * 4 : end * 4])
            current = tuple(item for item in identity if item[0] < end)
            identities[end] = current
            proof = identity_bytes(current)
            keyed = digest.copy()
            if proof:
                keyed.update(_IDENTITY_DOMAIN)
                keyed.update(struct.pack("<Q", len(proof)))
                keyed.update(proof)
            keys[end] = keyed.digest()
        return Plan(packed, keys, 0, identities=identities)

    def preview_cached_tokens(self, tokens, step=CHUNK, *, extra_keys=None,
                              extra_key_token_start=None, extra_key_ranges=None) -> int:
        """Hint at reusable tokens without restoring, waiting or changing LRU.

        An SSD catalog hit is only a hint. The scheduler must reprice the
        actual residual after prepare() and cache validation before execution.
        """
        try:
            plan = self._lookup_plan(tokens, step, extra_keys=extra_keys,
                                     extra_key_token_start=extra_key_token_start,
                                     extra_key_ranges=extra_key_ranges)
        except (TypeError, ValueError, OverflowError, struct.error):
            return 0
        if plan is None:
            return 0
        for end, key in reversed(list(plan.keys.items())):
            entry = self.entries.get(key)
            if entry is not None:
                if (type(entry.size) is int and 0 < entry.size <= self.max_bytes
                        and entry.tokens == plan.tokens[:end * 4]
                        and entry.identity == plan.identities[end]):
                    return end
            elif (self._l2 is not None
                  and self._l2.has_committed_checkpoint(key, end)):
                return end
        return 0

    def prepare(self, tokens, step=CHUNK, *, extra_keys=None,
                extra_key_token_start=None, extra_key_ranges=None):
        plan = self._lookup_plan(tokens, step, extra_keys=extra_keys,
                                 extra_key_token_start=extra_key_token_start,
                                 extra_key_ranges=extra_key_ranges)
        if plan is None:
            return None, 0, None
        packed, keys, identities = plan.tokens, plan.keys, plan.identities
        for end, key in reversed(list(keys.items())):
            entry = self.entries.get(key)
            from_l2 = entry is None
            if from_l2:
                if self._l2 is None:
                    continue
                if self.policy == FIBONACCI_TAIL and not self._l2.has_checkpoint(
                    key, count_miss=True
                ):
                    continue
            prefix = packed[: end * 4]
            if from_l2:
                entry = self._l2.load(key, prefix, identities[end])
            # Hashes are an index only: compare every token before reuse.
            if (entry is not None and entry.tokens == prefix
                    and entry.identity == identities[end]):
                t0 = time.perf_counter()
                self._protected_keys.add(key)
                try:
                    # Copies initially share storage. Reserve their eventual
                    # copy-on-write allocation before restoring any handles.
                    if not self._headroom(entry.size):
                        continue
                    cache = clone_cache(entry.cache)
                except Exception:
                    self.restore_failures += 1
                    logger.exception("DS41 prefix restore failed; using cold prefill")
                    continue
                finally:
                    self._protected_keys.discard(key)
                if not from_l2 or self.ram_admits_checkpoint(end):
                    if key not in self.entries:
                        # A pressure callback may clear the entire manager.
                        continue
                    self.entries.move_to_end(key)
                    self._revision += 1
                self.hits += 1
                if from_l2:
                    self._l2.confirm_hit(end)
                    self.l2_hits += 1
                else:
                    self.ram_hits += 1
                    if self._l2 is not None:
                        self._l2.touch(key)
                self.reused_tokens += end
                logger.info(
                    "DS41 exact prefix hit: tokens=%d restore_ms=%.3f hot_MiB=%.1f",
                    end,
                    (time.perf_counter() - t0) * 1000,
                    self.bytes / 2**20,
                )
                plan.next_position = end
                return cache, end, plan
        return None, 0, plan

    def record(self, plan, cache, start, count):
        if plan is None or not plan.valid:
            if plan is not None:
                self.record_skipped_invalid_plan += 1
            return
        if count != CHUNK or start != plan.next_position or start % CHUNK:
            if count != CHUNK:
                self.record_skipped_partial_chunks += 1
            else:
                self.record_skipped_position += 1
            plan.valid = False
            return
        end = start + count
        plan.next_position = end
        key = plan.keys.get(end)
        if key is None:
            self.record_skipped_key += 1
            plan.valid = False
            return
        if self.is_tail_checkpoint(end):
            last_boundary = ((len(plan.tokens) // 4 - 1) // CHUNK) * CHUNK
            if end != last_boundary:
                self.record_skipped_policy += 1
                return
        if key in self.entries:
            self.entries.move_to_end(key)
            self._revision += 1
            if self._l2 is not None:
                self._l2.store(key, self.entries[key], fresh=True)
            return
        arrays = cache_arrays(cache)
        # Include metadata/tokens and array views conservatively (shared buffers
        # may be double-counted). Evict BEFORE retaining another checkpoint.
        identity = plan.identities[end]
        size = sum(x.nbytes for x in arrays) + end * 4 + len(identity_bytes(identity))
        if size > self.max_bytes:
            self.record_skipped_oversized += 1
            return
        if not self.ram_admits_checkpoint(end):
            self.ssd_only_store_attempts += 1
            # The L2 stager consumes this request's state synchronously before
            # giving the worker immutable CPU bytes. No RAM snapshot is needed.
            if self._l2 is None or not self._l2.store(
                key, Entry(plan.tokens[: end * 4], cache, size, identity), fresh=True
            ):
                self.ssd_only_store_skipped += 1
            return
        self.evict_bytes(self.bytes + size - self.max_bytes)
        # A non-contiguous view can require both a compact temporary and its
        # owned snapshot. Existing active allocations already include source.
        if not self._headroom(2 * size):
            return
        mx.eval(arrays)
        snapshot = clone_cache(cache, compact=True)
        mx.eval(cache_arrays(snapshot))
        self.entries[key] = Entry(plan.tokens[: end * 4], snapshot, size, identity)
        self.bytes += size
        self.stores += 1
        self._revision += 1
        if self._l2 is not None:
            self._l2.store(key, self.entries[key], fresh=True)


def for_model(model):
    if not model._config.ced_prefill:
        return None
    limit = float(os.environ.get("DS41_PREFIX_CACHE_GIB", "0"))
    if limit <= 0:
        return None
    # Keep checkpoints outside nn.Module's parameter tree.
    signature = (
        bool(getattr(model, "_omlx_dspark_decode_enabled", False)),
        getattr(model, "_omlx_mtp_depth", None),
    )
    manager = model.__dict__.get("_exact_prefix_cache")
    if manager is None or getattr(manager, "signature", None) != signature:
        manager = ExactPrefixCache(
            min(limit, 3.0) * 2**30,
            image_token_id=getattr(model._config, "image_token_id", IMAGE_TOKEN_ID),
            policy=os.environ.get("DS41_EXACT_PREFIX_CHECKPOINT_POLICY", "dense"),
        )
        manager.signature = signature
        object.__setattr__(model, "_exact_prefix_cache", manager)
    return manager
