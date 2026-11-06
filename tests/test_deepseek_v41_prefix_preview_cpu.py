"""Cache admission previews inspect metadata only; no MLX or model is loaded."""

import ast
from collections import OrderedDict
import copy
from dataclasses import dataclass
import hashlib
import json
import logging
from pathlib import Path
import struct
import sys
import threading
import time
from types import SimpleNamespace as NS

import pytest


SOURCE = Path(__file__).resolve().parents[1] / "omlx/patches/deepseek_v41"
CHUNK = 8192


def forbidden(*args, **kwargs):
    pytest.fail("preview must not restore, wait, reserve memory or publish statistics")


@pytest.fixture(scope="module")
def definitions():
    before = set(sys.modules)
    namespace = dict(
        __name__=__name__, OrderedDict=OrderedDict, dataclass=dataclass,
        copy=copy, hashlib=hashlib, json=json, logging=logging, struct=struct,
        threading=threading, time=time, logger=logging.getLogger(__name__),
        CHUNK=CHUNK, IMAGE_TOKEN_ID=129264, FIBONACCI_TAIL="fibonacci_tail",
        FIBONACCI_RAM_MAX_TOKENS=21 * CHUNK, FIBONACCI_TAIL_MAX_BYTES=10 * 2**30,
        _IDENTITY_DOMAIN=b"\0ds41-image-prefix-identity-v1\0",
        MAX_CHECKPOINT_BYTES=3 * 2**30, clone_cache=forbidden,
    )
    selected = {
        "_is_fibonacci_checkpoint", "_identity_keys", "prefix_identity",
        "identity_bytes", "checkpoint_key", "_request_identity", "Plan", "Entry",
        "ExactPrefixCache", "ExactPrefixL2",
    }
    for filename in ("prefix_cache.py", "prefix_cache_l2.py"):
        tree = ast.parse((SOURCE / filename).read_text())
        nodes = [node for node in tree.body if getattr(node, "name", None) in selected]
        # Execute the production definitions unchanged, without package imports
        # that initialize MLX. Preview itself needs only Python metadata.
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE / filename), "exec"),
             namespace)
    assert not any(name == "mlx" or name.startswith("mlx.")
                   for name in set(sys.modules) - before)
    return NS(ns=namespace, cache=namespace["ExactPrefixCache"],
              l2=namespace["ExactPrefixL2"], entry=namespace["Entry"],
              key=namespace["checkpoint_key"])


def prompt(length):
    return [i % 100000 for i in range(length)]


def checkpoint(definitions, cache, tokens, end, identity=()):
    packed = struct.pack(f"<{end}I", *tokens[:end])
    key = definitions.key(packed, identity)
    entry = definitions.entry(packed, [object()], 1024, identity)
    cache.entries[key] = entry
    cache.bytes += entry.size
    return key, entry


def attach_l2(definitions, cache):
    l2 = definitions.l2.__new__(definitions.l2)
    l2.manager = cache
    l2.max_bytes = 8 * 2**30
    l2._closed = False
    l2._lock = threading.RLock()
    l2._pending_path = None
    l2.entries = OrderedDict()
    l2.hits = l2.misses = l2.reads = 0
    l2.load = l2._finish = l2.touch = l2.stats = forbidden
    cache._l2 = l2
    return l2


def catalog_row(key, end):
    # Existing L2 metadata has no image-identity field. The key is a hint until
    # prepare() validates the complete token bytes and the checkpoint identity.
    return dict(key=key.hex(), tokens=end, size=1024,
                file_bytes=2048, atime_ns=1234)


def test_ram_preview_leaves_final_token_and_reports_longest_complete_prefix(definitions):
    cache = definitions.cache(2**20)
    tokens = prompt(3 * CHUNK + 464)
    for end in (CHUNK, 2 * CHUNK, 3 * CHUNK):
        checkpoint(definitions, cache, tokens, end)
    assert cache.preview_cached_tokens(tokens) == 3 * CHUNK
    assert cache.preview_cached_tokens(tokens[:3 * CHUNK]) == 2 * CHUNK
    assert cache.preview_cached_tokens(tokens[:CHUNK]) == 0
    assert cache.preview_cached_tokens(tokens, step=512) == 0


def test_ram_preview_does_not_change_lru_stats_or_memory_state(definitions):
    cache = definitions.cache(2**20)
    tokens = prompt(2 * CHUNK + 464)
    # The longest hit starts at the LRU front, so moving it would be observable.
    checkpoint(definitions, cache, tokens, 2 * CHUNK)
    checkpoint(definitions, cache, tokens, CHUNK)
    attach_l2(definitions, cache)
    cache.ensure_headroom = forbidden
    cache.snapshot_stats = forbidden
    before = vars(cache).copy()
    entries = list(cache.entries.items())
    published = copy.deepcopy(cache._published_stats)
    assert cache.preview_cached_tokens(tokens) == 2 * CHUNK
    assert vars(cache) == before
    assert list(cache.entries.items()) == entries
    assert cache._published_stats == published
    assert not cache._protected_keys


@pytest.mark.parametrize("corruption", ["tokens", "identity", "zero_size", "oversize", "bool_size"])
def test_ram_preview_checks_complete_tokens_identity_and_size(definitions, corruption):
    cache = definitions.cache(2**20)
    tokens = prompt(2 * CHUNK + 464)
    checkpoint(definitions, cache, tokens, CHUNK)
    _, entry = checkpoint(definitions, cache, tokens, 2 * CHUNK)
    if corruption == "tokens":
        entry.tokens = entry.tokens[:-4] + struct.pack("<I", 7)
    elif corruption == "identity":
        entry.identity = ((0, ("a" * 64,)),)
    elif corruption == "zero_size":
        entry.size = 0
    elif corruption == "oversize":
        entry.size = cache.max_bytes + 1
    else:
        entry.size = True
    assert cache.preview_cached_tokens(tokens) == CHUNK


def test_image_preview_shares_prepare_keys_and_preserves_earlier_text_prefix(definitions, monkeypatch):
    cache = definitions.cache(2**20)
    tokens = prompt(2 * CHUNK + 464)
    tokens[CHUNK] = 129264
    checkpoint(definitions, cache, tokens, CHUNK)
    identity = ((CHUNK, ("a" * 64,)),)
    key, entry = checkpoint(definitions, cache, tokens, 2 * CHUNK, identity)
    metadata = dict(extra_keys=("a" * 64,), extra_key_token_start=CHUNK,
                    extra_key_ranges=identity)
    assert cache.preview_cached_tokens(tokens, **metadata) == 2 * CHUNK
    restored = object()
    monkeypatch.setitem(definitions.ns, "clone_cache", lambda value: restored)
    actual, count, plan = cache.prepare(tokens, **metadata)
    assert actual is restored and count == 2 * CHUNK
    assert plan.next_position == count and plan.keys[count] == key
    assert plan.identities[count] == entry.identity
    assert cache.preview_cached_tokens(tokens, extra_keys=("b" * 64,),
                                       extra_key_token_start=CHUNK) == CHUNK
    assert cache.preview_cached_tokens(tokens) == 0
    assert cache.preview_cached_tokens(tokens, extra_keys=("bad",)) == 0


def test_l2_preview_reads_only_committed_catalog_without_lru_or_stats_changes(definitions):
    cache = definitions.cache(2**20)
    l2 = attach_l2(definitions, cache)
    tokens = prompt(3 * CHUNK + 464)
    keys = {}
    for end in (2 * CHUNK, CHUNK):
        key = definitions.key(struct.pack(f"<{end}I", *tokens[:end]))
        keys[end] = key
        l2.entries[key.hex()] = catalog_row(key, end)
    pending_key = definitions.key(struct.pack(f"<{3 * CHUNK}I", *tokens[:3 * CHUNK]))
    l2._pending_path = Path(f".{pending_key.hex()}.test.pending")
    before = copy.deepcopy(l2.entries)
    state = vars(l2).copy()
    assert cache.preview_cached_tokens(tokens) == 2 * CHUNK
    assert l2.entries == before and vars(l2) == state
    assert (l2.hits, l2.misses, l2.reads) == (0, 0, 0)
    l2.entries.clear()
    assert cache.preview_cached_tokens(tokens) == 0


@pytest.mark.parametrize("field,value", [
    ("key", "0" * 64), ("tokens", CHUNK - 1), ("tokens", True),
    ("size", 0), ("size", -1), ("size", True), ("size", 2**20 + 1),
    ("file_bytes", 0), ("file_bytes", True), ("file_bytes", None),
])
def test_l2_preview_rejects_unusable_catalog_metadata(definitions, field, value):
    cache = definitions.cache(2**20)
    l2 = attach_l2(definitions, cache)
    tokens = prompt(CHUNK + 464)
    key = definitions.key(struct.pack(f"<{CHUNK}I", *tokens[:CHUNK]))
    row = catalog_row(key, CHUNK)
    row[field] = value
    l2.entries[key.hex()] = row
    assert cache.preview_cached_tokens(tokens) == 0


def test_l2_preview_does_not_wait_for_a_writer_publishing_the_catalog(definitions):
    cache = definitions.cache(2**20)
    l2 = attach_l2(definitions, cache)
    tokens = prompt(CHUNK + 464)
    key = definitions.key(struct.pack(f"<{CHUNK}I", *tokens[:CHUNK]))
    l2.entries[key.hex()] = catalog_row(key, CHUNK)
    calls = []
    l2._lock = NS(acquire=lambda *, blocking: calls.append(blocking) or False,
                  release=forbidden)
    assert cache.preview_cached_tokens(tokens) == 0
    assert calls == [False]


def test_preview_is_a_hint_and_prepare_can_refuse_or_miss(definitions):
    cache = definitions.cache(2**20)
    tokens = prompt(CHUNK + 464)
    key, _ = checkpoint(definitions, cache, tokens, CHUNK)
    assert cache.preview_cached_tokens(tokens) == CHUNK
    cache.ensure_headroom = lambda amount: False
    restored, count, plan = cache.prepare(tokens)
    assert restored is None and count == plan.next_position == 0
    cache.ensure_headroom = None
    cache.entries.pop(key)
    restored, count, plan = cache.prepare(tokens)
    assert restored is None and count == plan.next_position == 0


def test_disabled_preview_and_malformed_token_input_fail_closed(definitions):
    cache = definitions.cache(2**20)
    l2 = attach_l2(definitions, cache)
    tokens = prompt(CHUNK + 464)
    key = definitions.key(struct.pack(f"<{CHUNK}I", *tokens[:CHUNK]))
    l2.entries[key.hex()] = catalog_row(key, CHUNK)
    assert cache.preview_cached_tokens(tokens) == CHUNK
    l2._closed = True
    assert cache.preview_cached_tokens(tokens) == 0
    l2._closed, l2.max_bytes = False, 0
    assert cache.preview_cached_tokens(tokens) == 0
    l2.max_bytes, cache.max_bytes = 2**20, 0
    assert cache.preview_cached_tokens(tokens) == 0
    cache.max_bytes = 2**20
    tokens[-1] = -1
    assert cache.preview_cached_tokens(tokens) == 0
