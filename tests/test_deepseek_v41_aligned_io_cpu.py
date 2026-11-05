"""Real bounded file I/O and extracted loader contracts, without importing MLX."""
import ast
from contextlib import contextmanager
import errno
import hashlib
import importlib.util
import json
import mmap
import os
from pathlib import Path
import struct
from threading import Condition, Lock, RLock
from types import SimpleNamespace

import pytest

PATCH = Path(__file__).resolve().parents[1] / "omlx/patches/deepseek_v41"
SPEC = importlib.util.spec_from_file_location("aligned_io_under_test", PATCH / "aligned_io.py")
aligned = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(aligned)


def payload(size):
    return (bytes(range(251)) * ((size + 250) // 251))[:size]


@pytest.mark.parametrize("size", [0, 1, 16383, 16384, 16385, 8 * 1024**2 + 37])
def test_roundtrip_and_final_padding_removed(tmp_path, size):
    path = tmp_path / "exact.bin"
    expected = payload(size)
    with aligned.AlignedWriter(path) as writer:
        step = 7919
        for start in range(0, size, step):
            assert writer.write(memoryview(expected)[start:start + step]) == len(expected[start:start + step])
        assert writer.tell() == size
        writer.flush()
        assert path.stat().st_size == size
        assert writer.io_metrics["workspace_address_mod_alignment"] == 0
        assert writer.io_metrics["max_syscall_bytes"] <= aligned.IO_WORKSPACE_BYTES
    assert path.read_bytes() == expected
    with aligned.AlignedReader(path) as reader:
        assert reader.read() == expected
        assert reader.read(1) == b""
        assert reader.tell() == size
        assert reader.io_metrics["workspace_address_mod_alignment"] == 0
        assert reader.io_metrics["max_syscall_bytes"] <= aligned.IO_WORKSPACE_BYTES


def test_flush_then_append_and_multiple_partial_flushes(tmp_path):
    path = tmp_path / "append.bin"
    expected = b""
    with aligned.AlignedWriter(path) as writer:
        for size in (3, 16385, 91, 8 * 1024**2 + 29):
            part = payload(size)
            writer.write(part)
            expected += part
            writer.flush()
            assert path.read_bytes() == expected
            assert path.stat().st_size == writer.tell()
    assert path.read_bytes() == expected


def test_flush_fsync_close_does_not_rewrite_tail(tmp_path):
    path = tmp_path / "durable.bin"
    writer = aligned.AlignedWriter(path)
    writer.write(payload(16387))
    writer.flush()
    os.fsync(writer.fileno())
    calls = writer.io_metrics["syscalls"]
    writer.flush()
    assert writer.io_metrics["syscalls"] == calls
    writer.close()
    assert writer.io_metrics["syscalls"] == calls
    assert path.read_bytes() == payload(16387)


@pytest.mark.parametrize("offset,length", [(404, 37), (15137, 16384), (16383, 8 * 1024**2 + 19), (33, 2**20 + 19)])
def test_unaligned_seek_and_readinto_exact(tmp_path, offset, length):
    path = tmp_path / "source.bin"
    expected = payload(offset + length + 17)
    path.write_bytes(expected)
    target = bytearray(length)
    with aligned.AlignedReader(path) as reader:
        assert reader.seek(offset) == offset
        assert reader.readinto(target) == length
        assert bytes(target) == expected[offset:offset + length]
        assert reader.tell() == offset + length
        assert reader.seek(-17, os.SEEK_END) == len(expected) - 17
        assert reader.read(99) == expected[-17:]


def test_borrowed_fd_offset_and_ownership_preserved(tmp_path):
    path = tmp_path / "borrow.bin"
    path.write_bytes(payload(40000))
    with path.open("rb") as original:
        original.seek(23)
        fd_position = os.lseek(original.fileno(), 0, os.SEEK_CUR)
        with aligned.AlignedReader.from_fd(original.fileno(), closefd=False) as reader:
            reader.seek(15137)
            assert reader.read(41) == payload(40000)[15137:15178]
        assert os.lseek(original.fileno(), 0, os.SEEK_CUR) == fd_position
        assert original.read(3) == payload(40000)[23:26]


def test_negative_seek_and_closed_operations_rejected(tmp_path):
    path = tmp_path / "closed.bin"
    path.write_bytes(b"abc")
    reader = aligned.AlignedReader(path)
    with pytest.raises(ValueError):
        reader.seek(-1)
    with pytest.raises(ValueError):
        reader.seek(1, 99)
    with pytest.raises(TypeError):
        reader.readinto(b"readonly")
    reader.close()
    assert reader.closed
    with pytest.raises(ValueError):
        reader.read(1)
    with pytest.raises(ValueError):
        reader.fileno()


def test_truncation_after_open_fails_closed(tmp_path):
    path = tmp_path / "truncate.bin"
    path.write_bytes(payload(40000))
    with aligned.AlignedReader(path) as reader:
        os.truncate(path, 11)
        with pytest.raises(OSError, match="shortened"):
            reader.readinto(bytearray(40000))


def test_pread_eintr_and_partial_reads_are_exact(tmp_path, monkeypatch):
    path = tmp_path / "interrupt.bin"
    expected = payload(65537)
    path.write_bytes(expected)
    real_pread = aligned._LIBC.pread
    calls = 0
    def partial(fd, address, count, offset):
        nonlocal calls
        calls += 1
        if calls == 1:
            aligned.ctypes.set_errno(errno.EINTR)
            return -1
        # One real short read still contains the requested head; subsequent
        # aligned envelopes may reread that page and must preserve exact bytes.
        return real_pread(fd, address, min(count, 32768) if calls == 2 else count, offset)
    monkeypatch.setattr(aligned._LIBC, "pread", partial)
    with aligned.AlignedReader(path) as reader:
        reader.seek(404)
        assert reader.read() == expected[404:]
    assert calls >= 3


def test_writer_refuses_short_unaligned_retry(tmp_path, monkeypatch):
    path = tmp_path / "short.bin"
    writer = aligned.AlignedWriter(path)
    monkeypatch.setattr(aligned._LIBC, "pwrite", lambda *args: 1)
    try:
        writer.write(b"abc")
        with pytest.raises(OSError, match="Short aligned write"):
            writer.flush()
    finally:
        writer._dispose()


def native_read_into():
    tree = ast.parse((PATCH / "native_metadata.py").read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "NativeMetadataPlan")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_read_into")
    namespace = {"mx": SimpleNamespace(default_device=lambda: "cpu", gpu="gpu"),
                 "np": SimpleNamespace(asarray=lambda value: value.data),
                 "AlignedReader": aligned.AlignedReader,
                 "IO_WORKSPACE_BYTES": aligned.IO_WORKSPACE_BYTES,
                 "_IO_BYTES": 8 * 1024**2}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(PATCH / "native_metadata.py"), "exec"), namespace)
    return namespace["_read_into"]


def test_native_read_into_digest_excludes_alignment_padding(tmp_path):
    source = tmp_path / "native.bin"
    prefix, expected, suffix = b"P" * 15137, payload(8 * 1024**2 + 113), b"S" * 167
    source.write_bytes(prefix + expected + suffix)
    checks = []
    plan = SimpleNamespace(_check=lambda count, stage: checks.append((count, stage)))
    value = SimpleNamespace(size=len(expected), data=bytearray(len(expected)))
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        native_read_into()(plan, stream, len(prefix), value, digest, "fixture")
        assert not stream.closed
    assert value.data == expected
    assert digest.hexdigest() == hashlib.sha256(expected).hexdigest()
    assert checks[0] == (aligned.IO_WORKSPACE_BYTES, "aligned I/O workspace fixture")
    assert checks[1:] == [(0, "read fixture"), (0, "read fixture")]


def test_native_workspace_guard_refuses_before_arena_allocation(tmp_path, monkeypatch):
    source = tmp_path / "refuse.bin"
    source.write_bytes(b"abc")
    def refuse(*args):
        raise RuntimeError("physical cap")
    def must_not_allocate(*args, **kwargs):
        raise AssertionError("arena allocated before guard")
    monkeypatch.setattr(aligned.AlignedReader, "from_fd", must_not_allocate)
    with source.open("rb") as stream, pytest.raises(RuntimeError, match="physical cap"):
        native_read_into()(SimpleNamespace(_check=refuse), stream, 0,
                           SimpleNamespace(size=3, data=bytearray(3)), None, "fixture")


def test_native_tensorfile_skips_mapping_and_engram_default_keeps_it(tmp_path):
    tree = ast.parse((PATCH / "storage.py").read_text())
    original = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "TensorFile")
    cls = ast.ClassDef(name="TensorFile", bases=[], keywords=[], decorator_list=[],
                       body=[node for node in original.body if isinstance(node, ast.FunctionDef)
                             and node.name in ("__init__", "close")])
    namespace = dict(RLock=RLock, Path=Path, Condition=Condition, Lock=Lock,
                     struct=struct, json=json, mmap=mmap, os=os)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), "storage_contract", "exec"), namespace)
    header = json.dumps({"x": {"dtype": "U8", "shape": [3], "data_offsets": [0, 3]}}).encode()
    source = tmp_path / "fixture.safetensors"
    source.write_bytes(struct.pack("<Q", len(header)) + header + b"abc")
    reader = namespace["TensorFile"](source, map_file=False)
    assert reader._mapping is None
    assert reader.header["x"]["data_offsets"] == [0, 3]
    reader.close()
    engram = namespace["TensorFile"](source)
    assert engram._mapping is not None
    engram.close()
