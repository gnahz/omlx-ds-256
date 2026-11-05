"""Exercise exact-cache disk I/O and admission without importing MLX."""

import ast
from collections import OrderedDict
import hashlib
import importlib.util
import json
import logging
import math
import os
from pathlib import Path
import struct
import sys
import threading
import time
from types import SimpleNamespace as NS

import numpy as np
import pytest

SOURCE = Path(__file__).resolve().parents[1] / "omlx/patches/deepseek_v41"


@pytest.fixture(scope="module")
def definitions():
    before = set(sys.modules)
    spec = importlib.util.spec_from_file_location("cache_aligned_io_cpu", SOURCE / "aligned_io.py")
    io = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(io)
    namespace = dict(
        __name__=__name__, hashlib=hashlib, json=json, logging=logging, math=math, os=os,
        Path=Path, struct=struct, OrderedDict=OrderedDict, time=time, np=np,
        logger=logging.getLogger(__name__), IO_WORKSPACE_BYTES=io.IO_WORKSPACE_BYTES,
        AlignedReader=io.AlignedReader, AlignedWriter=io.AlignedWriter,
        MAX_METADATA=8 * 2**20, MAX_CHECKPOINT_BYTES=3 * 2**30,
        FORMAT="ds41-exact-prefix-v3-ced-replay",
        identity_bytes=lambda identity: b"[]", prefix_identity=lambda identity, end: (),
        _DTYPES={"I32": (None, 4), "U32": (None, 4)},
    )
    selected = {"_json_bytes", "_integer", "_write_checkpoint", "_validate_file",
                "_load_checkpoint", "PrefixCacheStore", "ExactPrefixL2"}
    for filename in ("prefix_cache_store.py", "prefix_cache_l2.py"):
        tree = ast.parse((SOURCE / filename).read_text())
        nodes = [node for node in tree.body if getattr(node, "name", None) in selected]
        # Runtime classes are injected below; importing their package would load
        # MLX. The executable function and method bodies remain unchanged.
        for node in nodes:
            for child in ast.walk(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    child.body = [item for item in child.body if not isinstance(item, ast.ImportFrom)]
        exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                     str(SOURCE / filename), "exec"), namespace)
    assert not any(name == "mlx" or name.startswith("mlx.") for name in set(sys.modules) - before)
    return NS(io=io, ns=namespace, l2=namespace["ExactPrefixL2"],
              persistence=namespace["PrefixCacheStore"])


def checkpoint(definitions, directory):
    ns = definitions.ns
    tokens = np.arange(8192, dtype="<u4").tobytes()
    chunks = [struct.pack("<i", 8192), tokens]
    layout = {"layers": [{"compress_ratio": None, "meta_state": ["deepseek_v41", "2"],
                           "slots": ["offset"] + [None] * 6,
                           "left_padding": None, "lengths": None}], "prime": None}
    table = {"offset": {"dtype": "I32", "shape": [1], "data_offsets": [0, 4]},
             "tokens": {"dtype": "U32", "shape": [8192], "data_offsets": [4, 4 + len(tokens)]},
             "__metadata__": {"format": ns["FORMAT"],
                              "layout": ns["_json_bytes"](layout).decode(), "identity": "[]"}}
    encoded = ns["_json_bytes"](table)
    encoded += b" " * (-len(encoded) % 8)
    header = struct.pack("<Q", len(encoded)) + encoded
    descriptor = dict(key="a" * 64, tokens=8192, size=sum(map(len, chunks)) + 2,
                      identity=(), layout=layout)
    path = directory / "checkpoint.safetensors"
    written = ns["_write_checkpoint"](path, header, chunks, descriptor)
    return path, written, header + b"".join(chunks)


def test_checkpoint_writer_preserves_every_byte_checksum_and_exact_tail(definitions, tmp_path):
    path, descriptor, expected = checkpoint(definitions, tmp_path)
    assert path.read_bytes() == expected
    assert path.stat().st_size == descriptor["file_bytes"] == len(expected)
    assert descriptor["sha256"] == hashlib.sha256(expected).hexdigest()
    assert len(expected) % 16384 != 0


def test_validator_uses_aligned_reader_and_rejects_corruption(definitions, tmp_path, monkeypatch):
    path, descriptor, _ = checkpoint(definitions, tmp_path)
    calls = []
    real = definitions.io.AlignedReader
    def tracked(path):
        calls.append(path)
        return real(path)
    monkeypatch.setitem(definitions.ns, "AlignedReader", tracked)
    assert definitions.ns["_validate_file"](path, descriptor, 2**20) == descriptor["size"]
    assert calls == [path]
    with path.open("r+b") as stream:
        stream.seek(-1, os.SEEK_END)
        stream.write(b"\xff")
    with pytest.raises(ValueError, match="checksum"):
        definitions.ns["_validate_file"](path, descriptor, 2**20)


def test_mlx_loader_receives_open_aligned_stream_and_evaluates_before_close(definitions, tmp_path, monkeypatch):
    path, descriptor, _ = checkpoint(definitions, tmp_path)
    calls, streams = [], []
    arrays = {"offset": np.array([8192], dtype=np.int32),
              "tokens": np.arange(8192, dtype="<u4")}
    def load(stream, **kwargs):
        assert not stream.closed and not isinstance(stream, (str, Path))
        assert kwargs == {"format": "safetensors"}
        streams.append(stream)
        return arrays.copy()
    def evaluate(values):
        calls.append(streams[0].closed)
    monkeypatch.setitem(definitions.ns, "mx", NS(load=load, eval=evaluate))
    monkeypatch.setitem(definitions.ns, "cache_arrays", lambda cache: [cache[0].cache[0]])
    monkeypatch.setitem(definitions.ns, "DeepseekV41Cache", lambda ratio: NS(compress_ratio=ratio))
    monkeypatch.setitem(definitions.ns, "checkpoint_key", lambda tokens, identity: bytes.fromhex(descriptor["key"]))
    monkeypatch.setitem(definitions.ns, "Entry", lambda tokens, cache, size, identity: NS(tokens=tokens, cache=cache))
    result = definitions.ns["_load_checkpoint"](path, descriptor)
    assert result.cache[0].cache[0].item() == 8192
    assert calls[0] is False and streams[0].closed


def l2_instance(definitions, directory, accepted):
    item = definitions.l2.__new__(definitions.l2)
    calls = []
    answers = iter(accepted)
    def headroom(amount):
        calls.append(amount)
        return next(answers)
    item.manager = NS(_protected_keys=set(), _headroom=headroom, max_bytes=2**20,
                      bytes=0, evict_bytes=lambda size: None)
    item.directory, item._closed, item._staging = directory, False, False
    item.max_bytes, item._cleared_keys = 2**20, set()
    item._lock, item._finish = threading.RLock(), lambda **kwargs: True
    item.entries = {"a" * 64: {"file_bytes": 100}}
    item.skipped_memory = item.misses = item.reads = item.read_bytes = 0
    item.write_failures = 0
    return item, calls


def test_catalog_probe_counts_only_enabled_l2_misses(definitions, tmp_path):
    item, calls = l2_instance(definitions, tmp_path, [])
    indexed, missing, pending = (bytes.fromhex(char * 64) for char in "abc")
    item._pending_path = tmp_path / f".{pending.hex()}.test.pending"

    assert item.has_checkpoint(indexed, count_miss=True)
    assert item.has_checkpoint(pending, count_miss=True)
    assert not item.has_checkpoint(missing)
    assert item.misses == 0
    assert not item.has_checkpoint(missing, count_miss=True)
    assert item.misses == 1

    item._closed = True
    assert not item.has_checkpoint(missing, count_miss=True)
    assert item.misses == 1 and not calls


def test_l2_workspace_refusal_precedes_payload_validation_and_retains_disk_entry(definitions, tmp_path, monkeypatch):
    item, calls = l2_instance(definitions, tmp_path, [False])
    monkeypatch.setitem(definitions.ns, "_validate_file", lambda *args: pytest.fail("payload read before admission"))
    assert item.load(bytes.fromhex("a" * 64), b"", ()) is None
    assert calls == [definitions.io.IO_WORKSPACE_BYTES]
    assert item.skipped_memory == item.misses == 1
    assert "a" * 64 in item.entries and not item._staging


def test_l2_load_preserves_two_snapshot_charge_and_adds_reader_workspace(definitions, tmp_path, monkeypatch):
    item, calls = l2_instance(definitions, tmp_path, [True, False])
    monkeypatch.setitem(definitions.ns, "_validate_file", lambda *args: 1234)
    monkeypatch.setitem(definitions.ns, "_load_checkpoint", lambda *args, **kwargs: pytest.fail("allocated after refusal"))
    assert item.load(bytes.fromhex("a" * 64), b"", ()) is None
    assert calls == [definitions.io.IO_WORKSPACE_BYTES, 2468 + definitions.io.IO_WORKSPACE_BYTES]
    assert "a" * 64 in item.entries


def test_l2_writer_admits_staging_and_workspace_before_staging(definitions, tmp_path, monkeypatch):
    item, calls = l2_instance(definitions, tmp_path, [False])
    item.entries.clear()
    monkeypatch.setitem(definitions.ns, "_stage", lambda *args: pytest.fail("staged after refusal"))
    assert not item.store(bytes.fromhex("a" * 64), NS(size=1234))
    assert calls == [2468 + definitions.io.IO_WORKSPACE_BYTES]
    assert item.skipped_memory == 1 and not item.manager._protected_keys


def test_restart_workspace_refusal_preserves_durable_catalog_without_advertising_hit(definitions, tmp_path, monkeypatch):
    item = definitions.persistence.__new__(definitions.persistence)
    calls = []
    item._restore_attempted = False
    item.manager = NS(_headroom=lambda amount: calls.append(amount) or False,
                      entries=OrderedDict(), bytes=0, _revision=0, _protected_keys=set())
    item.restore_skipped = item.disk_reads = item.restored_entries = 0
    item.directory = tmp_path
    manifest = {"generation": "generation-" + "b" * 32,
                "entries": [{"key": "a" * 64}]}
    item._manifest = lambda name: manifest
    monkeypatch.setitem(definitions.ns, "_validate_file", lambda *args: pytest.fail("payload read after refusal"))
    assert item.restore() == 0
    assert calls == [definitions.io.IO_WORKSPACE_BYTES]
    assert item.restore_skipped == 1 and item.disk_reads == 0
    assert item._committed_manifest == manifest and not item.manager.entries


def test_restart_writer_preserves_two_snapshot_charge_plus_workspace(definitions, tmp_path, monkeypatch):
    item = definitions.persistence.__new__(definitions.persistence)
    calls = []
    item._closed, item.future, item.future_kind = False, None, None
    item.transaction = {"keys": [b"a"], "index": 0}
    item.manager = NS(entries={b"a": NS(size=1234)}, _protected_keys=set(),
                      _headroom=lambda amount: calls.append(amount) or False)
    item.failures = 0
    monkeypatch.setitem(definitions.ns, "_stage", lambda *args: pytest.fail("staged after refusal"))
    assert not item.step(force=True)
    assert calls == [2468 + definitions.io.IO_WORKSPACE_BYTES]
    assert item.failures == 1 and not item.manager._protected_keys
