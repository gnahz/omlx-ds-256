# SPDX-License-Identifier: MIT
"""Byte-preserving, bounded uncached file I/O for immutable model/cache files.

Darwin F_NOCACHE alone still takes the cached path for unaligned file offsets.
Read an aligned envelope into one small aligned arena and copy the requested
bytes. Sequential writes use the same alignment and truncate final padding.
Each open instance owns at most IO_WORKSPACE_BYTES of arena storage.
"""

import ctypes
import errno
import fcntl
import io
import os


IO_ALIGNMENT = 16 * 1024
IO_CHUNK_BYTES = 8 * 1024**2
IO_WORKSPACE_BYTES = IO_CHUNK_BYTES + 2 * IO_ALIGNMENT
_LIBC = ctypes.CDLL(None, use_errno=True)
_LIBC.posix_memalign.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t,
                               ctypes.c_size_t]
_LIBC.posix_memalign.restype = ctypes.c_int
_LIBC.free.argtypes = [ctypes.c_void_p]
_LIBC.pread.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int64]
_LIBC.pread.restype = ctypes.c_ssize_t
_LIBC.pwrite.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int64]
_LIBC.pwrite.restype = ctypes.c_ssize_t


def _aligned_up(value):
    return (value + IO_ALIGNMENT - 1) // IO_ALIGNMENT * IO_ALIGNMENT


class _AlignedFile:
    def _initialize(self, fd, *, closefd, name):
        self._fd = fd
        self._closefd = closefd
        self.name = name
        self._closed = False
        self._arena = ctypes.c_void_p()
        try:
            fcntl.fcntl(fd, getattr(fcntl, "F_NOCACHE", 48), 1)
            fcntl.fcntl(fd, getattr(fcntl, "F_RDAHEAD", 45), 0)
            error = _LIBC.posix_memalign(ctypes.byref(self._arena), IO_ALIGNMENT,
                                          IO_WORKSPACE_BYTES)
            if error:
                raise OSError(error, os.strerror(error))
            self._storage = (ctypes.c_ubyte * IO_WORKSPACE_BYTES).from_address(self._arena.value)
            self._view = memoryview(self._storage).cast("B")
            self.io_metrics = {"alignment_bytes": IO_ALIGNMENT,
                               "workspace_bytes": IO_WORKSPACE_BYTES,
                               "workspace_address_mod_alignment": self._arena.value % IO_ALIGNMENT,
                               "syscalls": 0, "max_syscall_bytes": 0,
                               "logical_bytes": 0, "physical_bytes": 0,
                               "max_read_return_bytes": 0}
        except BaseException:
            self._dispose()
            raise

    @property
    def closed(self):
        return self._closed

    def _require_open(self):
        if self._closed:
            raise ValueError("I/O operation on closed aligned file")

    def fileno(self):
        self._require_open()
        return self._fd

    def _dispose(self):
        if getattr(self, "_closed", True):
            return
        self._closed = True
        view = getattr(self, "_view", None)
        if view is not None:
            view.release()
            del self._view, self._storage
        if self._arena.value:
            _LIBC.free(self._arena)
            self._arena = ctypes.c_void_p()
        if self._closefd:
            os.close(self._fd)

    def __enter__(self):
        self._require_open()
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        # Explicit close/context use is required; cleanup must not raise at exit.
        try:
            self._dispose()
        except Exception:
            pass


class AlignedReader(_AlignedFile):
    def __init__(self, path):
        self._initialize(os.open(path, os.O_RDONLY), closefd=True, name=os.fspath(path))
        self._position = 0
        self._size = os.fstat(self._fd).st_size

    @classmethod
    def from_fd(cls, fd, closefd=False):
        result = cls.__new__(cls)
        result._initialize(fd, closefd=closefd, name=f"<fd:{fd}>")
        result._position = 0
        result._size = os.fstat(fd).st_size
        return result

    def readable(self):
        return not self.closed

    def seekable(self):
        return not self.closed

    def writable(self):
        return False

    def tell(self):
        self._require_open()
        return self._position

    def seek(self, offset, whence=os.SEEK_SET):
        self._require_open()
        if whence == os.SEEK_CUR:
            offset += self._position
        elif whence == os.SEEK_END:
            offset += self._size
        elif whence != os.SEEK_SET:
            raise ValueError("Invalid seek mode")
        if offset < 0:
            raise ValueError("Negative seek position")
        self._position = offset
        return offset

    def readinto(self, target):
        self._require_open()
        target = memoryview(target).cast("B")
        if target.readonly:
            raise TypeError("readinto requires a writable buffer")
        remaining = min(len(target), max(0, self._size - self._position))
        done = 0
        while done < remaining:
            base = self._position // IO_ALIGNMENT * IO_ALIGNMENT
            head = self._position - base
            wanted = min(remaining - done, IO_CHUNK_BYTES)
            envelope = _aligned_up(head + wanted)
            while True:
                got = _LIBC.pread(self._fd, self._arena, envelope, base)
                if got >= 0:
                    break
                error = ctypes.get_errno()
                if error != errno.EINTR:
                    raise OSError(error, os.strerror(error), self.name)
            available = min(wanted, max(0, got - head))
            self.io_metrics["syscalls"] += 1
            self.io_metrics["max_syscall_bytes"] = max(self.io_metrics["max_syscall_bytes"], envelope)
            self.io_metrics["physical_bytes"] += got
            if available == 0:
                raise OSError("Source file shortened during aligned read")
            target[done:done + available] = self._view[head:head + available]
            self._position += available
            done += available
            self.io_metrics["logical_bytes"] += available
        return done

    readinto1 = readinto

    def read(self, size=-1):
        self._require_open()
        if size is None or size < 0:
            size = max(0, self._size - self._position)
        size = min(size, max(0, self._size - self._position))
        self.io_metrics["max_read_return_bytes"] = max(self.io_metrics["max_read_return_bytes"], size)
        result = bytearray(size)
        self.readinto(result)
        return bytes(result)

    def close(self):
        self._dispose()


class AlignedWriter(_AlignedFile):
    """Sequential writer; flush/close keep final on-disk bytes exactly unpadded."""

    def __init__(self, path):
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o666)
        self._initialize(fd, closefd=True, name=os.fspath(path))
        self._committed = 0
        self._buffered = 0
        self._dirty = False

    def writable(self):
        return not self.closed

    def readable(self):
        return False

    def seekable(self):
        return False

    def tell(self):
        self._require_open()
        return self._committed + self._buffered

    def seek(self, *args):
        raise io.UnsupportedOperation("AlignedWriter is sequential")

    def _write_aligned(self, count):
        while True:
            wrote = _LIBC.pwrite(self._fd, self._arena, count, self._committed)
            if wrote >= 0:
                break
            error = ctypes.get_errno()
            if error != errno.EINTR:
                raise OSError(error, os.strerror(error), self.name)
        if wrote != count:
            raise OSError("Short aligned write; refusing an unaligned retry")
        self.io_metrics["syscalls"] += 1
        self.io_metrics["max_syscall_bytes"] = max(self.io_metrics["max_syscall_bytes"], count)
        self.io_metrics["physical_bytes"] += wrote

    def write(self, source):
        self._require_open()
        source = memoryview(source).cast("B")
        if source:
            self._dirty = True
        done = 0
        while done < len(source):
            take = min(len(source) - done, IO_CHUNK_BYTES - self._buffered)
            self._view[self._buffered:self._buffered + take] = source[done:done + take]
            self._buffered += take
            done += take
            if self._buffered == IO_CHUNK_BYTES:
                self._write_aligned(self._buffered)
                self._committed += self._buffered
                self._buffered = 0
        self.io_metrics["logical_bytes"] += done
        return done

    def flush(self):
        self._require_open()
        if not self._dirty:
            return
        full = self._buffered // IO_ALIGNMENT * IO_ALIGNMENT
        if full:
            self._write_aligned(full)
            self._committed += full
            self._buffered -= full
            if self._buffered:
                ctypes.memmove(self._arena, self._arena.value + full, self._buffered)
        if self._buffered:
            padded = _aligned_up(self._buffered)
            self._view[self._buffered:padded] = bytes(padded - self._buffered)
            self._write_aligned(padded)
        os.ftruncate(self._fd, self.tell())
        self._dirty = False

    def close(self):
        if not self.closed:
            try:
                self.flush()
            finally:
                self._dispose()
