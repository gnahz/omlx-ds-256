# SPDX-License-Identifier: MIT
"""Bounded, on-demand SSD backing for complete exact CED checkpoints.

The engine owns all MLX work. One worker receives only immutable CPU bytes;
there is never a queue of retained MLX checkpoints. The index becomes visible
only after the complete safetensors file has been fsynced and renamed.
"""

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import copy
import hashlib
import logging
from pathlib import Path
import os
import re
import threading
import time
import uuid

from .aligned_io import IO_WORKSPACE_BYTES
from .prefix_cache_store import (
    MAX_CHECKPOINT_BYTES, MAX_METADATA, _atomic_json, _fsync_directory, _integer,
    _json_bytes, _load_checkpoint, _read_json, _stage, _validate_file,
    _write_checkpoint,
)

logger = logging.getLogger(__name__)
FORMAT = "deepseek-v41-exact-l2-v1"
BACKEND = "deepseek_v41_exact_l2_v1"
OWNER = ".omlx-deepseek-exact-l2-v1"


def _key(value):
    if (not isinstance(value, str) or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)):
        raise ValueError("invalid L2 checkpoint key")
    return value


class ExactPrefixL2:
    def __init__(self, manager, directory, identity, max_bytes):
        self.manager = manager
        self._owner_thread = manager._owner_thread
        self._published_stats = {}
        self.directory = Path(directory)
        self.max_bytes = _integer(max_bytes)
        self.identity = identity
        self.fingerprint = hashlib.sha256(_json_bytes(identity)).hexdigest()
        self.entries = OrderedDict()
        self._lock = threading.RLock()
        self._epoch = 0
        self._closed = False
        self._staging = False
        self._touch_dirty = False
        self._cleared_keys = set()
        self._future = None
        self._pending_bytes = 0
        self._pending_tail_bytes = 0
        self._pending_path = None
        self.last_error = None
        self.hits = self.misses = self.reads = self.writes = 0
        self.fixed_hits = self.tail_hits = self.ssd_only_hits = 0
        self.read_bytes = self.write_bytes = self.evictions = 0
        self.read_failures = self.write_failures = self.skipped_memory = 0
        self.invalidations = 0
        self.directory.mkdir(parents=True, exist_ok=True)
        if self.directory.is_symlink() or not self.directory.is_dir():
            raise ValueError("L2 directory must be a real directory")
        self._check_owner()
        self._load_index()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ds41-exact-l2")
        self.stats()

    @property
    def index_path(self):
        return self.directory / "index.json"

    @property
    def tail_max_bytes(self):
        maximum = getattr(self.manager, "tail_max_bytes", None)
        return None if maximum is None else min(maximum, self.max_bytes)

    def _tail_sizes(self):
        if self.tail_max_bytes is None:
            return {}
        return {key: row["file_bytes"] for key, row in self.entries.items()
                if self.manager.is_tail_checkpoint(row["tokens"])}

    def _manifest(self, entries=None):
        return dict(format=FORMAT, identity=self.identity, fingerprint=self.fingerprint,
                    entries=list((self.entries if entries is None else entries).values()))

    def _check_owner(self):
        marker = self.directory / OWNER
        if not marker.exists():
            if any(self.directory.iterdir()):
                raise ValueError("refusing nonempty unowned L2 directory")
            marker.touch(exist_ok=False)
            _fsync_directory(self.directory)
        if marker.is_symlink() or not marker.is_file() or marker.stat().st_size:
            raise ValueError("invalid L2 ownership marker")
        for path in self.directory.iterdir():
            allowed = path.name in (OWNER, "index.json", "index.json.tmp")
            allowed |= bool(re.fullmatch(r"[0-9a-f]{64}\.safetensors", path.name))
            allowed |= bool(re.fullmatch(r"\.[0-9a-f]{64}\.[0-9a-f]{32}\.pending", path.name))
            if not allowed or (path.is_dir() and not path.is_symlink()):
                raise ValueError("refusing unknown contents in L2 directory")

    def _disk_bytes(self):
        # Include the index in the quota, and separately reserve the complete
        # pending payload plus its atomic replacement index before submission.
        total = 0
        for path in self.directory.iterdir():
            # Active temporary files are covered by their full reservation.
            # Failed/orphan files remain charged even if absent from index.
            if path == self._pending_path and self._pending_bytes:
                continue
            if path.name == "index.json.tmp" and self._pending_bytes:
                continue
            if path.is_file() and not path.is_symlink():
                total += path.stat().st_size
        return total

    def _save_index(self):
        if self.index_path.is_symlink():
            raise ValueError("L2 index must not be a symlink")
        try:
            _atomic_json(self.index_path, self._manifest())
            _fsync_directory(self.directory)
        except Exception:
            try:
                self.index_path.with_name("index.json.tmp").unlink(missing_ok=True)
            except OSError:
                pass
            raise

    def _load_index(self):
        try:
            if self.index_path.is_symlink():
                raise ValueError("L2 index must not be a symlink")
            if self.index_path.exists():
                value = _read_json(self.index_path)
                stored_fingerprint = hashlib.sha256(_json_bytes(value.get("identity"))).hexdigest()
                if value.get("format") != FORMAT or value.get("fingerprint") != stored_fingerprint:
                    raise ValueError("invalid L2 deployment identity")
                if stored_fingerprint != self.fingerprint:
                    # A deliberate model/draft/runtime change invalidates only
                    # this owned tier; it is not a corrupt-file incident.
                    self.invalidations += 1
                    rows = []
                else:
                    rows = value["entries"]
                if not isinstance(rows, list) or len(rows) > 100000:
                    raise ValueError("invalid L2 index")
                for row in sorted(rows, key=lambda item: _integer(item["atime_ns"])):
                    key = _key(row["key"])
                    _integer(row["size"], 1, MAX_CHECKPOINT_BYTES)
                    _integer(row["tokens"], 8192, MAX_CHECKPOINT_BYTES // 4)
                    amount = _integer(row["file_bytes"], 1)
                    path = self.directory / (key + ".safetensors")
                    if key in self.entries or path.is_symlink():
                        raise ValueError("duplicate or symlink L2 checkpoint")
                    if path.is_file() and path.stat().st_size == amount:
                        self.entries[key] = row
        except Exception as error:
            self.entries.clear()
            self.read_failures += 1
            self.last_error = f"{type(error).__name__}: {error}"
        # Only files owned by this dedicated backend can be garbage-collected.
        # Do not follow links or recursively remove arbitrary directories.
        for path in self.directory.iterdir():
            if path.name.endswith(".pending") or path.name == "index.json.tmp":
                if path.is_file() or path.is_symlink():
                    path.unlink()
            elif path.suffix == ".safetensors":
                try:
                    name = _key(path.stem)
                except ValueError:
                    continue
                if name not in self.entries and (path.is_file() or path.is_symlink()):
                    path.unlink()
        self._trim(0)

    def _evict(self, key):
        # A quota decision must not make idle writeback immediately
        # resurrect the same RAM entry and evict another in a loop.
        # A subsequent request use/record makes it eligible again.
        binary_key = bytes.fromhex(key)
        if self.tail_max_bytes is None or binary_key in self.manager.entries:
            self._cleared_keys.add(binary_key)
        path = self.directory / (key + ".safetensors")
        if path.is_file() or path.is_symlink():
            path.unlink()
        del self.entries[key]
        self.evictions += 1

    def _trim(self, reservation, *, tail_reservation=0):
        """Reserve data AND atomic-index bytes before opening a new file."""
        tail_limit = self.tail_max_bytes
        if tail_limit is not None:
            tails = self._tail_sizes()
            tail_bytes = sum(tails.values())
            # Avoid evicting useful tails if they cannot make this write fit.
            if tail_reservation and (
                tail_reservation > tail_limit
                or self._disk_bytes() - tail_bytes + reservation > self.max_bytes
            ):
                return False
            for key, size in tails.items():
                if (tail_bytes + tail_reservation <= tail_limit
                        and self._disk_bytes() + reservation <= self.max_bytes):
                    break
                self._evict(key)
                tail_bytes -= size
            if tail_reservation:
                return (tail_bytes + tail_reservation <= tail_limit
                        and self._disk_bytes() + reservation <= self.max_bytes)
        while self.entries and self._disk_bytes() + reservation > self.max_bytes:
            self._evict(next(iter(self.entries)))
        if self._disk_bytes() + reservation > self.max_bytes:
            # Empty metadata must not defeat a zero/very small quota.
            if not self.entries and self.index_path.exists():
                self.index_path.unlink()
            if self._disk_bytes() + reservation > self.max_bytes:
                return False
        return True

    def _finish(self, wait=False, timeout_seconds=None):
        with self._lock:
            future = self._future
            if future is None or (not wait and not future.done()):
                return future is None
        error = None
        try:
            future.result(timeout=timeout_seconds)
        except TimeoutError as caught:
            if not future.done():
                return False
            error = caught
        except Exception as caught:
            error = caught
        # Waiting must not hold the catalog lock needed by the worker. An
        # accidental concurrent poll may retire the same completed future;
        # never clear a newer write or count the same failure twice.
        with self._lock:
            if self._future is not future:
                return self._future is None
            if error is not None:
                self.write_failures += 1
                self.last_error = f"{type(error).__name__}: {error}"
                logger.warning("DS41 exact L2 write failed: %s", error)
            self._future = None
            self._pending_bytes = 0
            self._pending_tail_bytes = 0
            self._pending_path = None
            return True

    def _write_one(self, temporary, header, chunks, descriptor, epoch):
        key = descriptor["key"]
        try:
            result = _write_checkpoint(temporary, header, chunks, descriptor)
            with self._lock:
                if self._closed or epoch != self._epoch:
                    return
                target = self.directory / (key + ".safetensors")
                if target.is_symlink():
                    raise ValueError("L2 checkpoint must not be a symlink")
                os.replace(temporary, target)
                _fsync_directory(self.directory)
                self.entries[key] = result
                self.entries.move_to_end(key)
                try:
                    self._save_index()
                except Exception:
                    self.entries.pop(key, None)
                    target.unlink(missing_ok=True)
                    raise
                self.writes += 1
                self._touch_dirty = False
                self.write_bytes += result["file_bytes"]
                self.last_error = None
                self._pending_bytes = 0
                self._pending_tail_bytes = 0
        finally:
            temporary.unlink(missing_ok=True)

    def store(self, key, entry, *, fresh=False):
        """Prewrite or spill with one CPU buffer and explicit engine headroom."""
        if self._closed or not self.max_bytes or self._staging:
            return False
        if fresh:
            self._cleared_keys.discard(key)
        if key in self._cleared_keys or entry.size > self.max_bytes:
            return False
        if not self._finish(wait=True, timeout_seconds=5):
            return False
        name = key.hex()
        with self._lock:
            if name in self.entries:
                self.touch(key)
                return True
        self._staging = True
        already_protected = key in self.manager._protected_keys
        self.manager._protected_keys.add(key)
        try:
            if not self.manager._headroom(2 * entry.size + IO_WORKSPACE_BYTES):
                self.skipped_memory += 1
                return False
            header, chunks, layout = _stage(entry)
            descriptor = dict(key=name, tokens=len(entry.tokens) // 4,
                              size=entry.size, identity=entry.identity, layout=layout,
                              atime_ns=time.time_ns())
            file_bytes = len(header) + sum(map(len, chunks))
            tail_bytes = (
                file_bytes if self.tail_max_bytes is not None
                and self.manager.is_tail_checkpoint(descriptor["tokens"]) else 0
            )
            # Include checksum and file size in the future index estimate.
            estimated = dict(descriptor, sha256="0" * 64, file_bytes=file_bytes)
            with self._lock:
                prospective = OrderedDict(self.entries)
                prospective[name] = estimated
                index_bytes = len(_json_bytes(self._manifest(prospective)))
                if index_bytes > MAX_METADATA:
                    return False
                reservation = file_bytes + index_bytes
                if not self._trim(reservation, tail_reservation=tail_bytes):
                    return False
                # Publish removal metadata before any asynchronous write; an
                # interrupted write can never resurrect evicted catalog rows.
                if self.entries:
                    self._save_index()
                elif self.index_path.exists():
                    self.index_path.unlink()
                self._pending_bytes = reservation
                self._pending_tail_bytes = tail_bytes
                self._pending_path = self.directory / ("." + name + "." + uuid.uuid4().hex + ".pending")
                self._future = self.executor.submit(
                    self._write_one, self._pending_path, header, chunks, descriptor, self._epoch)
            return True
        except Exception as error:
            if self._future is None:
                self._pending_bytes = 0
                self._pending_tail_bytes = 0
                self._pending_path = None
            self.write_failures += 1
            self.last_error = f"{type(error).__name__}: {error}"
            logger.warning("DS41 exact L2 staging failed: %s", error)
            return False
        finally:
            if not already_protected:
                self.manager._protected_keys.discard(key)
            self._staging = False

    def has_committed_checkpoint(self, key, token_count):
        """Read only a published catalog row; never wait for or load a file."""
        if (self._closed or not self.max_bytes or type(token_count) is not int
                or not 8192 <= token_count <= MAX_CHECKPOINT_BYTES // 4):
            return False
        name = key.hex()
        # A writer holds this lock while publishing and syncing the index.
        # A busy catalog is an unknown hint, never a reason to wait for I/O.
        if not self._lock.acquire(blocking=False):
            return False
        try:
            row = self.entries.get(name)
            if not isinstance(row, dict):
                return False
            size, tokens = row.get("size"), row.get("tokens")
            file_bytes = row.get("file_bytes")
            return (row.get("key") == name
                    and type(tokens) is int and tokens == token_count
                    and type(size) is int
                    and 0 < size <= min(MAX_CHECKPOINT_BYTES, self.manager.max_bytes)
                    and type(file_bytes) is int and file_bytes > 0)
        finally:
            self._lock.release()

    def has_checkpoint(self, key, *, count_miss=False):
        """Check the catalog or a matching pending write without reading the disk."""
        if self._closed or not self.max_bytes:
            return False
        name = key.hex()
        with self._lock:
            pending = self._pending_path
            present = name in self.entries or (
                pending is not None and pending.name.startswith("." + name + ".")
            )
            if count_miss and not present:
                self.misses += 1
            return present

    def load(self, key, tokens, identity):
        """Load one validated candidate; never advertise incomplete/unsafe hits."""
        if self._closed or not self.max_bytes:
            return None
        if self.tail_max_bytes is not None and not self.has_checkpoint(key):
            self.misses += 1
            return None
        if not self._finish(wait=True, timeout_seconds=5):
            self.misses += 1
            return None
        name = key.hex()
        # Pressure callbacks can evict RAM entries. Never recursively stage
        # them and wait for a worker needing the catalog lock held below.
        self._staging = True
        with self._lock:
            row = self.entries.get(name)
            if row is None:
                self.misses += 1
                self._staging = False
                return None
            path = self.directory / (name + ".safetensors")
            try:
                if not self.manager._headroom(IO_WORKSPACE_BYTES):
                    self.skipped_memory += 1
                    self.misses += 1
                    return None
                self.reads += 1
                size = _validate_file(path, row, MAX_CHECKPOINT_BYTES)
                self.read_bytes += row["file_bytes"]
                if size > self.manager.max_bytes:
                    self.misses += 1
                    return None
                promote = (self.tail_max_bytes is None
                           or self.manager.ram_admits_checkpoint(row["tokens"]))
                if promote:
                    # Only resident checkpoints compete for the RAM quota.
                    self.manager.evict_bytes(self.manager.bytes + size - self.manager.max_bytes)
                    if self.manager.bytes + size > self.manager.max_bytes:
                        self.misses += 1
                        return None
                # mx.load plus the request's later copy-on-write clone can
                # coexist; reserve both before allocating either MLX state.
                if not self.manager._headroom(2 * size + IO_WORKSPACE_BYTES):
                    self.skipped_memory += 1
                    self.misses += 1
                    return None
                entry = _load_checkpoint(path, row, image_token_id=self.manager.image_token_id)
                self.read_bytes += row["file_bytes"]
                if entry.tokens != tokens or entry.identity != identity:
                    raise ValueError("L2 full-token or image identity mismatch")
                if promote:
                    self.manager.entries[key] = entry
                    self.manager.bytes += size
                    self.manager._revision += 1
                row["atime_ns"] = time.time_ns()
                self.entries.move_to_end(name)
                # Persist LRU order without exceeding the temporary-index
                # reservation. Dropping a cold copy leaves the RAM hit valid.
                try:
                    self._trim(len(_json_bytes(self._manifest())))
                    if self.entries:
                        self._save_index()
                    elif self.index_path.exists():
                        self.index_path.unlink()
                except Exception as error:
                    self.write_failures += 1
                    self.last_error = f"{type(error).__name__}: {error}"
                return entry
            except Exception as error:
                self.read_failures += 1
                self.misses += 1
                self.last_error = f"{type(error).__name__}: {error}"
                self.entries.pop(name, None)
                try:
                    if path.is_file() or path.is_symlink():
                        path.unlink()
                except Exception:
                    pass
                try:
                    self._save_index()
                except OSError:
                    pass
                logger.warning("DS41 exact L2 read rejected: %s", error)
                return None
            finally:
                self._staging = False

    def step(self):
        if not self._finish():
            return True
        self._flush_touches()
        if not self._closed and self.last_error is None:
            for key in self._unwritten_keys():
                entry = self.manager.entries.get(key)
                if entry is not None:
                    self.store(key, entry)
                    break
        return self._future is not None

    def _unwritten_keys(self):
        return [key for key in self.manager.entries
                if key.hex() not in self.entries and key not in self._cleared_keys]

    def touch(self, key):
        if self._closed:
            return
        with self._lock:
            # A new use may legitimately cache a RAM checkpoint again after
            # explicit SSD clear. An idle flush alone must not resurrect it.
            self._cleared_keys.discard(key)
            row = self.entries.get(key.hex())
            if row is not None:
                row["atime_ns"] = time.time_ns()
                self.entries.move_to_end(key.hex())
                self._touch_dirty = True

    def _flush_touches(self):
        if not self._touch_dirty or self._closed:
            return
        with self._lock:
            try:
                self._trim(len(_json_bytes(self._manifest())))
                if self.entries:
                    self._save_index()
                elif self.index_path.exists():
                    self.index_path.unlink()
                self._touch_dirty = False
            except Exception as error:
                self.write_failures += 1
                self.last_error = f"{type(error).__name__}: {error}"

    def confirm_hit(self, end=None):
        # Count only a checkpoint actually returned by ExactPrefixCache.prepare,
        # after the request's own clone/headroom checks have also succeeded.
        self.hits += 1
        if end is not None and self.tail_max_bytes is not None:
            if self.manager.is_tail_checkpoint(end):
                self.tail_hits += 1
            else:
                self.fixed_hits += 1
            if not self.manager.ram_admits_checkpoint(end):
                self.ssd_only_hits += 1

    def flush(self, timeout_seconds=None):
        # A stats/maintenance poll may already have consumed a failed future.
        # Do not subsequently relabel that failed write as a clean flush.
        deadline = time.monotonic() + (30 if timeout_seconds is None else max(0, timeout_seconds))
        attempted = set()
        while True:
            if not self._finish(wait=True, timeout_seconds=max(0, deadline - time.monotonic())):
                return False
            if self.last_error is not None:
                return False
            if self._closed:
                return True
            remaining = self._unwritten_keys()
            if not remaining:
                self._flush_touches()
                return self.last_error is None
            if time.monotonic() >= deadline:
                return False
            if remaining[0] in attempted:
                return False
            attempted.add(remaining[0])
            entry = self.manager.entries.get(remaining[0])
            if entry is not None and not self.store(remaining[0], entry):
                return False

    def set_max_bytes(self, max_bytes, timeout_seconds=30):
        maximum = _integer(max_bytes)
        self._epoch += 1
        if not self._finish(wait=True, timeout_seconds=timeout_seconds):
            raise TimeoutError("L2 writer did not quiesce for capacity change")
        with self._lock:
            previous = self.max_bytes
            self.max_bytes = maximum
            try:
                self._trim(0)
                if self.entries:
                    self._trim(len(_json_bytes(self._manifest())))
                if self._disk_bytes() > self.max_bytes:
                    raise OSError("cannot satisfy L2 capacity with retained files")
                if self.entries:
                    self._save_index()
                elif self.index_path.exists():
                    self.index_path.unlink()
            except Exception:
                self.max_bytes = previous
                raise

    def clear(self, timeout_seconds=30):
        self._epoch += 1
        # Suppress idle retry as soon as clear is requested, including a
        # timeout waiting for a writer that subsequently notices this epoch.
        self._cleared_keys.update(self.manager.entries)
        if not self._finish(wait=True, timeout_seconds=timeout_seconds):
            raise TimeoutError("L2 writer did not quiesce for clear")
        with self._lock:
            self._check_owner()
            self._cleared_keys.update(self.manager.entries)
            count = len(self.entries)
            for path in self.directory.iterdir():
                if path.name != OWNER and (path.is_file() or path.is_symlink()):
                    path.unlink()
            self.entries.clear()
            self.index_path.unlink(missing_ok=True)
            _fsync_directory(self.directory)
            return count

    def close(self, wait=True, timeout_seconds=None):
        success = self.flush(30 if timeout_seconds is None else timeout_seconds) if wait else self._future is None
        self._closed = True
        self._epoch += 1
        self.executor.shutdown(wait=wait and success, cancel_futures=True)
        return success

    def stats(self):
        if threading.get_ident() != self._owner_thread:
            return copy.deepcopy(self._published_stats)
        self._finish()
        with self._lock:
            tails = self._tail_sizes()
            result = dict(backend=BACKEND, enabled=bool(self.max_bytes) and not self._closed,
                        directory=str(self.directory), max_bytes=self.max_bytes,
                        disk_bytes=self._disk_bytes(), entries=len(self.entries),
                        hits=self.hits, misses=self.misses, reads=self.reads, writes=self.writes,
                        read_bytes=self.read_bytes, write_bytes=self.write_bytes,
                        evictions=self.evictions, read_failures=self.read_failures,
                        write_failures=self.write_failures, skipped_memory=self.skipped_memory,
                        invalidations=self.invalidations,
                        pending_bytes=self._pending_bytes, pending=self._future is not None,
                        tail_max_bytes=self.tail_max_bytes or 0, tail_entries=len(tails),
                        tail_disk_bytes=sum(tails.values()),
                        tail_pending_bytes=self._pending_tail_bytes,
                        fixed_entries=len(self.entries) - len(tails),
                        fixed_disk_bytes=sum(row["file_bytes"] for key, row in self.entries.items()
                                             if key not in tails),
                        tail_hits=self.tail_hits, fixed_hits=self.fixed_hits,
                        ssd_only_hits=self.ssd_only_hits,
                        unwritten_ram_checkpoints=len(self._unwritten_keys()),
                        unbacked_ram_checkpoints=sum(key.hex() not in self.entries
                                                    for key in self.manager.entries),
                        last_error=self.last_error)
        self._published_stats = copy.deepcopy(result)
        return result
