# SPDX-License-Identifier: MIT
"""Restart-only persistence for exact CED snapshots.

All tensor evaluation/conversion/reconstruction runs in the calling engine
thread. The sole worker receives immutable bytes and plain metadata only.
Files use the public safetensors format (no pickle or Python object loading).
"""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import shutil
import struct
import time
import uuid

import mlx.core as mx
import numpy as np

from .cache import DeepseekV41Cache
from .aligned_io import AlignedReader, AlignedWriter, IO_WORKSPACE_BYTES
from ..mlx_lm_mtp.deepseek_v4_dspark import DSparkContextCache, _DSparkPrimeContext

logger = logging.getLogger(__name__)
FORMAT = "ds41-exact-prefix-v3-ced-replay"
MAX_METADATA = 8 * 2**20
# Durable validation must not depend on this process's smaller memory budget.
# Files remain capped by the supported exact-cache pool size before any load.
MAX_CHECKPOINT_BYTES = 3 * 2**30
_DTYPES = {
    "BOOL": (mx.bool_, 1), "U8": (mx.uint8, 1), "I8": (mx.int8, 1),
    "U16": (mx.uint16, 2), "I16": (mx.int16, 2),
    "U32": (mx.uint32, 4), "I32": (mx.int32, 4),
    "U64": (mx.uint64, 8), "I64": (mx.int64, 8),
    "F16": (mx.float16, 2), "BF16": (mx.bfloat16, 2), "F32": (mx.float32, 4),
}


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _read_json(path):
    with path.open("rb") as stream:
        data = stream.read(MAX_METADATA + 1)
    if len(data) > MAX_METADATA:
        raise ValueError("oversized manifest")
    return json.loads(data)


def _atomic_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(_json_bytes(value))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _fsync_directory(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _integer(value, low=0, high=2**63 - 1):
    if type(value) is not int or not low <= value <= high:
        raise ValueError("invalid integer metadata")
    return value


def _stage(entry):
    """Make one CPU checkpoint, preserving raw BF16/NaN/ring bits."""
    from .prefix_cache import cache_arrays, identity_bytes

    mx.eval(cache_arrays(entry.cache))
    tensors, chunks, offset = {}, [], 0

    def add(name, value):
        nonlocal offset
        if value is None:
            return None
        dtype = next((name for name, (kind, _) in _DTYPES.items() if kind == value.dtype), None)
        if dtype is None:
            raise ValueError(f"unsupported checkpoint dtype {value.dtype}")
        raw = np.asarray(mx.contiguous(value).view(mx.uint8)).tobytes()
        tensors[name] = {"dtype": dtype, "shape": list(value.shape),
                         "data_offsets": [offset, offset + len(raw)]}
        chunks.append(raw)
        offset += len(raw)
        return name

    layers = []
    for index, layer in enumerate(entry.cache):
        prefix = f"layer.{index}."
        layers.append({
            "compress_ratio": layer.compress_ratio,
            "meta_state": list(layer.meta_state),
            "slots": [add(prefix + str(slot), value) for slot, value in enumerate(layer.cache)],
            "left_padding": add(prefix + "left_padding", layer.left_padding),
            "lengths": add(prefix + "lengths", layer.lengths),
        })
    ctx = getattr(entry.cache[0], "_omlx_mtp_prime_ctx", None)
    prime = None if ctx is None else {
        "expected_target_offset": ctx.expected_target_offset,
        "stages": [{"max_size": stage.max_size, "offset": stage.offset,
                    "keys": add(f"prime.{index}", stage.keys)}
                   for index, stage in enumerate(ctx.caches)],
    }
    tensors["tokens"] = {"dtype": "U32", "shape": [len(entry.tokens) // 4],
                         "data_offsets": [offset, offset + len(entry.tokens)]}
    chunks.append(entry.tokens)
    layout = {"layers": layers, "prime": prime}
    tensors["__metadata__"] = {"format": FORMAT, "layout": _json_bytes(layout).decode(),
                               "identity": identity_bytes(entry.identity).decode()}
    header = _json_bytes(tensors)
    header += b" " * (-len(header) % 8)
    if len(header) > MAX_METADATA:
        raise ValueError("oversized checkpoint header")
    return struct.pack("<Q", len(header)) + header, chunks, layout


def _write_checkpoint(path, header, chunks, descriptor):
    checksum = hashlib.sha256()
    with AlignedWriter(path) as stream:
        for data in [header, *chunks]:
            checksum.update(data)
            stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    return {**descriptor, "sha256": checksum.hexdigest(), "file_bytes": path.stat().st_size}


def _validate_file(path, descriptor, max_bytes):
    """Reject sizes/shapes/references before mx.load can allocate anything."""
    from .prefix_cache import identity_bytes, prefix_identity

    end = _integer(descriptor["tokens"], 8192, max_bytes // 4)
    if end % 8192 or len(descriptor["key"]) != 64:
        raise ValueError("invalid checkpoint token boundary")
    size = _integer(descriptor["size"], end * 4, max_bytes)
    identity = prefix_identity(descriptor["identity"], end)
    proof = identity_bytes(identity)
    payload_size = size - len(proof)
    if payload_size < end * 4:
        raise ValueError("invalid checkpoint image identity size")
    if path.is_symlink() or path.stat().st_size != descriptor["file_bytes"]:
        raise ValueError("invalid checkpoint file size")
    with AlignedReader(path) as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError("truncated checkpoint")
        length = struct.unpack("<Q", prefix)[0]
        if not 0 < length <= MAX_METADATA:
            raise ValueError("invalid safetensors header size")
        raw_header = stream.read(length)
        if len(raw_header) != length:
            raise ValueError("truncated header")
        tensors = json.loads(raw_header)
        if not isinstance(tensors, dict) or len(tensors) > 2048:
            raise ValueError("invalid tensor table")
        metadata = tensors.pop("__metadata__", None)
        if metadata != {"format": FORMAT, "layout": _json_bytes(descriptor["layout"]).decode(),
                        "identity": proof.decode()}:
            raise ValueError("checkpoint layout metadata mismatch")
        total = 0
        for spec in tensors.values():
            dtype = spec["dtype"]
            shape = spec["shape"]
            if dtype not in _DTYPES or not isinstance(shape, list) or len(shape) > 5:
                raise ValueError("invalid dtype or rank")
            for dim in shape:
                _integer(dim, 0, max_bytes)
            amount = math.prod(shape) * _DTYPES[dtype][1]
            start, stop = spec["data_offsets"]
            _integer(start, 0, payload_size)
            _integer(stop, start, payload_size)
            if stop - start != amount:
                raise ValueError("tensor shape does not match stored byte range")
            total += amount
        ranges = sorted((spec["data_offsets"] for spec in tensors.values()), key=lambda x: (x[0], x[1]))
        cursor = 0
        for start, stop in ranges:
            if start != cursor:
                raise ValueError("overlapping or sparse tensor offsets")
            cursor = stop
        if (cursor != payload_size or total != payload_size
                or 8 + length + payload_size != descriptor["file_bytes"]):
            raise ValueError("checkpoint exceeds declared memory size")
        if tensors.get("tokens") != {"dtype": "U32", "shape": [end],
                                     "data_offsets": [payload_size - end * 4, payload_size]}:
            raise ValueError("invalid token tensor")
        layout = descriptor["layout"]
        layers = layout["layers"]
        if not isinstance(layers, list) or not 1 <= len(layers) <= 128:
            raise ValueError("invalid layer count")
        names = {"tokens"}

        def reference(name, rank, integer=False):
            if name is None:
                return None
            if not isinstance(name, str) or name in names or name not in tensors:
                raise ValueError("invalid tensor reference")
            names.add(name)
            spec = tensors[name]
            if len(spec["shape"]) != rank or spec["shape"][0] != 1:
                raise ValueError("invalid batch/rank")
            if integer and spec["dtype"] not in ("I32", "U32", "I64", "U64"):
                raise ValueError("invalid integer state dtype")
            return spec

        for layer in layers:
            ratio = layer["compress_ratio"]
            if ratio is not None:
                _integer(ratio, 0, 8192)
            slots = layer["slots"]
            meta = layer.get("meta_state")
            expected_meta = (["deepseek_v41", "2"] if ratio is None else
                             ["deepseek_v41", "4" if len(slots) == 11 else "3", str(ratio)])
            if len(slots) not in (7, 11) or meta != expected_meta:
                raise ValueError("invalid layer state")
            if len(slots) == 11 and (ratio is None or any(name is None for name in slots[7:])):
                raise ValueError("invalid decoder replay state")
            for slot, name in enumerate(slots):
                rank = 1 if slot == 0 else 2 if slot == 6 else 4 if slot == 7 else 3
                reference(name, rank, integer=slot in (0, 6, 9, 10))
            if len(slots) == 11:
                hidden, pre = (tensors[slots[i]] for i in (7, 8))
                if (hidden["dtype"] not in ("F16", "BF16", "F32")
                        or pre["dtype"] != "F32" or hidden["shape"][:3] != pre["shape"]
                        or not 0 < hidden["shape"][1] <= end):
                    raise ValueError("invalid decoder replay geometry")
            if layer["slots"][0] is None:
                raise ValueError("missing layer offset")
            reference(layer["left_padding"], 1, integer=True)
            reference(layer["lengths"], 1, integer=True)
        prime = layout["prime"]
        if prime is not None:
            if _integer(prime["expected_target_offset"]) != end:
                raise ValueError("invalid prime context position")
            stages = prime["stages"]
            if not isinstance(stages, list) or not 1 <= len(stages) <= 128:
                raise ValueError("invalid draft stage count")
            for stage in stages:
                maximum = _integer(stage["max_size"], 1, 2**24)
                if _integer(stage["offset"]) != end:
                    raise ValueError("invalid draft offset")
                spec = reference(stage["keys"], 4)
                if spec is not None and spec["shape"][2] > min(maximum, end):
                    raise ValueError("invalid draft ring size")
        if names != set(tensors):
            raise ValueError("unreferenced tensors")
        checksum = hashlib.sha256(prefix + raw_header)
        while data := stream.read(2**20):
            checksum.update(data)
        if checksum.hexdigest() != descriptor["sha256"]:
            raise ValueError("checkpoint checksum mismatch")
    return size


def _load_checkpoint(path, descriptor, *, image_token_id=129264):
    from .prefix_cache import Entry, cache_arrays, checkpoint_key, prefix_identity

    # File-like loading evaluates before returning; keep the aligned reader
    # alive through all evaluation rather than handing MLX a cached path reader.
    with AlignedReader(path) as stream:
        arrays = mx.load(stream, format="safetensors")
        mx.eval(list(arrays.values()))
    tokens = np.asarray(arrays.pop("tokens")).astype("<u4", copy=False).tobytes()
    identity = prefix_identity(descriptor["identity"], descriptor["tokens"])
    ids = np.frombuffer(tokens, dtype="<u4")
    image_positions = np.flatnonzero(ids == image_token_id)
    if image_positions.size and (not identity or identity[0][0] > int(image_positions[0])):
        raise ValueError("checkpoint image tokens have no content identity")
    if checkpoint_key(tokens, identity).hex() != descriptor["key"]:
        raise ValueError("checkpoint token digest mismatch")
    cache = []
    get = lambda name: None if name is None else arrays[name]
    for layer in descriptor["layout"]["layers"]:
        item = DeepseekV41Cache(layer["compress_ratio"])
        item.cache = [get(name) for name in layer["slots"]]
        item.meta_state = layer["meta_state"]
        item.left_padding, item.lengths = get(layer["left_padding"]), get(layer["lengths"])
        if int(item.cache[0].item()) != descriptor["tokens"]:
            raise ValueError("checkpoint target offset mismatch")
        cache.append(item)
    prime = descriptor["layout"]["prime"]
    if prime is not None:
        stages = []
        for state in prime["stages"]:
            stage = DSparkContextCache(state["max_size"])
            stage.offset, stage.keys = state["offset"], get(state["keys"])
            stages.append(stage)
        cache[0]._omlx_mtp_prime_ctx = _DSparkPrimeContext(stages, prime["expected_target_offset"])
    mx.eval(cache_arrays(cache))
    return Entry(tokens, cache, descriptor["size"], identity)


class PrefixCacheStore:
    def __init__(self, manager, directory, fingerprint, interval_seconds):
        self.manager = manager
        self.directory = Path(directory)
        self.fingerprint = hashlib.sha256(_json_bytes(fingerprint)).hexdigest()
        self.identity = fingerprint
        self.interval = max(0, float(interval_seconds))
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ds41-cache-ssd")
        self.future = None
        self.future_kind = None
        self.transaction = None
        self.persisted_revision = -1
        self.last_save = time.monotonic()
        self.saved_at = None
        self.saved_generation = None
        self.restored_entries = self.restore_skipped = self.saves = self.failures = 0
        self.last_error = None
        self._closed = False
        self._restore_attempted = False
        self._committed_manifest = None
        self.disk_reads = self.restore_loads = 0

    @property
    def pending(self):
        return self.future is not None or self.transaction is not None

    def stats(self):
        return dict(enabled=True, directory=str(self.directory), pending=self.pending,
                    interval_seconds=self.interval, saved_at=self.saved_at,
                    saved_generation=self.saved_generation,
                    disk_reads=self.disk_reads, restore_loads=self.restore_loads,
                    saves=self.saves, restored_checkpoints=self.restored_entries,
                    restore_skipped=self.restore_skipped, failures=self.failures,
                    last_error=self.last_error,
                    dirty=self.manager._revision != self.persisted_revision)

    def _manifest(self, name):
        manifest = _read_json(self.directory / name)
        if manifest["format"] != FORMAT or manifest["fingerprint"] != self.fingerprint:
            raise ValueError("checkpoint model/runtime/config fingerprint mismatch")
        generation = manifest["generation"]
        if (not isinstance(generation, str) or not generation.startswith("generation-")
                or len(generation) != 43 or any(c not in "0123456789abcdef" for c in generation[11:])):
            raise ValueError("invalid checkpoint generation")
        if not isinstance(manifest["entries"], list) or len(manifest["entries"]) > 16384:
            raise ValueError("invalid checkpoint list")
        keys = [entry["key"] for entry in manifest["entries"]]
        if len(set(keys)) != len(keys) or any(
            not isinstance(key, str) or len(key) != 64
            or any(c not in "0123456789abcdef" for c in key) for key in keys
        ):
            raise ValueError("invalid or duplicate checkpoint keys")
        return manifest

    def restore(self):
        if self._restore_attempted:
            return 0
        self._restore_attempted = True
        manager = self.manager
        for name in ("current.json", "previous.json"):
            try:
                manifest = self._manifest(name)
            except FileNotFoundError:
                continue
            except Exception as exc:
                self.last_error = str(exc)
                self.restore_skipped += 1
                continue
            restored = []
            verified = []
            deferred = []
            # Newest first under the memory cap; insert back in original LRU order.
            for descriptor in reversed(manifest["entries"]):
                try:
                    key = bytes.fromhex(descriptor["key"])
                    path = self.directory / manifest["generation"] / (key.hex() + ".safetensors")
                    manager._protected_keys.update(restored)
                    try:
                        if not manager._headroom(IO_WORKSPACE_BYTES):
                            # Preserve the durable catalog when validation cannot
                            # allocate its bounded I/O workspace. This is a retained
                            # disk reference only; it is never a RAM cache hit.
                            deferred.append(descriptor)
                            self.restore_skipped += 1
                            continue
                    finally:
                        manager._protected_keys.difference_update(restored)
                    self.disk_reads += 1
                    size = _validate_file(path, descriptor, MAX_CHECKPOINT_BYTES)
                    verified.append(descriptor)
                    if (not manager.ram_admits_checkpoint(descriptor["tokens"])
                            or manager.bytes + size > manager.max_bytes):
                        self.restore_skipped += 1
                        continue
                    manager._protected_keys.update(restored)
                    try:
                        if not manager._headroom(2 * size + IO_WORKSPACE_BYTES):
                            self.restore_skipped += 1
                            continue
                    finally:
                        manager._protected_keys.difference_update(restored)
                    self.disk_reads += 1
                    self.restore_loads += 1
                    entry = _load_checkpoint(path, descriptor,
                                             image_token_id=manager.image_token_id)
                    manager.entries[key] = entry
                    manager.bytes += entry.size
                    restored.append(key)
                except Exception as exc:
                    logger.warning("Skipping DS41 persisted checkpoint: %s", exc)
                    self.last_error = str(exc)
                    self.restore_skipped += 1
            if verified or deferred or not manifest["entries"]:
                # A pressure callback may have evicted earlier restored entries.
                manager.entries = type(manager.entries)((key, manager.entries[key])
                    for key in reversed(restored) if key in manager.entries)
                self.restored_entries = len(manager.entries)
                self.saved_at = manifest.get("saved_at")
                self.saved_generation = manifest["generation"]
                retained = {row["key"] for row in verified + deferred}
                self._committed_manifest = {**manifest, "entries": [
                    row for row in manifest["entries"] if row["key"] in retained
                ]}
                manager._revision += 1
                self.persisted_revision = manager._revision
                logger.info("DS41 startup cache restore: checkpoints=%d bytes=%d skipped=%d",
                            self.restored_entries, manager.bytes, self.restore_skipped)
                return self.restored_entries
        return 0

    def _begin(self, transaction):
        self.directory.mkdir(parents=True, exist_ok=True)
        # The engine passes the last startup-verified or freshly committed
        # manifest. Never reread checkpoint payloads during ordinary operation.
        current = transaction["previous_manifest"]
        keep = {current["generation"]} if current is not None else set()
        # Failed initial writes have no manifest yet. They are still orphans
        # and must be reclaimed before another attempt creates a generation.
        for directory in self.directory.glob("generation-*"):
            if directory.is_dir() and directory.name not in keep:
                shutil.rmtree(directory, ignore_errors=True)
        if current is not None:
            # Keep an atomic pointer to the verified generation even if the
            # current manifest was corrupt at startup and we used the backup.
            _atomic_json(self.directory / "previous.json", current)
        (self.directory / transaction["generation"]).mkdir()

    def _commit(self, transaction):
        manifest = dict(format=FORMAT, fingerprint=self.fingerprint, identity=self.identity,
                        generation=transaction["generation"], entries=transaction["entries"],
                        saved_at=time.time())
        old = transaction["previous_manifest"]
        _fsync_directory(self.directory / transaction["generation"])
        if old is not None:
            _atomic_json(self.directory / "previous.json", old)
            _fsync_directory(self.directory)
        _atomic_json(self.directory / "current.json", manifest)
        _fsync_directory(self.directory)
        keep = {manifest["generation"]}
        if old:
            keep.add(old["generation"])
        for directory in self.directory.glob("generation-*"):
            if directory.is_dir() and directory.name not in keep:
                shutil.rmtree(directory, ignore_errors=True)
        return manifest

    def step(self, force=False):
        if self._closed:
            return False
        try:
            if self.future is not None:
                if not self.future.done():
                    return True
                result = self.future.result()
                self.future = None
                if self.future_kind == "commit":
                    self.saved_at = result["saved_at"]
                    self.saved_generation = self.transaction["generation"]
                    self._committed_manifest = result
                    self.persisted_revision = self.transaction["revision"]
                    self.transaction = None
                    self.last_save = time.monotonic()
                    self.saves += 1
                    self.last_error = None
                    return force and self.manager._revision != self.persisted_revision
                if self.future_kind == "write":
                    self.transaction["entries"].append(result)
                self.future_kind = None
            if self.transaction is None:
                if self.manager._revision == self.persisted_revision:
                    return False
                if not force and time.monotonic() - self.last_save < self.interval:
                    return False
                self.transaction = dict(generation="generation-" + uuid.uuid4().hex,
                    keys=list(self.manager.entries), entries=[],
                    revision=self.manager._revision, index=0,
                    previous_manifest=self._committed_manifest)
                self.future_kind = "begin"
                self.future = self.executor.submit(self._begin, self.transaction)
                return True
            transaction = self.transaction
            while transaction["index"] < len(transaction["keys"]):
                key = transaction["keys"][transaction["index"]]
                transaction["index"] += 1
                entry = self.manager.entries.get(key)
                if entry is None:
                    continue
                self.manager._protected_keys.add(key)
                try:
                    if not self.manager._headroom(2 * entry.size + IO_WORKSPACE_BYTES):
                        raise RuntimeError("insufficient memory to stage checkpoint; save deferred")
                    header, chunks, layout = _stage(entry)
                finally:
                    self.manager._protected_keys.discard(key)
                descriptor = dict(key=key.hex(), tokens=len(entry.tokens) // 4,
                                  size=entry.size, layout=layout, identity=entry.identity)
                path = self.directory / transaction["generation"] / (key.hex() + ".safetensors")
                self.future_kind = "write"
                self.future = self.executor.submit(_write_checkpoint, path, header, chunks, descriptor)
                return True
            transaction["index"] = "commit"
            self.future_kind = "commit"
            self.future = self.executor.submit(self._commit, transaction)
            return True
        except Exception as exc:
            self.last_error = str(exc)
            self.failures += 1
            self.last_save = time.monotonic()
            self.future = None
            self.transaction = None
            logger.exception("DS41 checkpoint persistence failed; keeping committed snapshot")
            return False

    def close(self, wait=True):
        self._closed = True
        # Shutdown callers first drain step(force=True) on the engine thread.
        # A bounded caller can close(wait=False); only one raw-byte write can remain.
        self.executor.shutdown(wait=wait, cancel_futures=True)
