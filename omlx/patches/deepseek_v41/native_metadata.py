# SPDX-License-Identifier: MIT
"""Opt-in native G128 scale/bias dictionaries; original 3-bit codes are unchanged.

The existing DwarfStar files contain one U16 index per original BF16 pair and
a U32 dictionary of the two raw BF16 words. Loading excludes the original
scales/biases before any shard is read. Decode looks up pairs in registers;
prefill can optionally look up pairs in shared weight tiles; the fallback
expands and drains only one expert bank at a time. This module never
encodes or requantizes weights and never drops draft or vision components.

QMV arithmetic is from Apple's MIT-licensed MLX 0.32.2 quantized.h and the
qualified DwarfStar metal/mlx_oQ3e.metal. Only metadata addressing changes.
The independent DS41_NATIVE_METADATA flag takes a metadata-proof.json path;
an unset flag leaves the ordinary loader unchanged.
"""

from dataclasses import dataclass, replace
from functools import cache
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import struct

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .quantization import QuantizedProjection, quantize_activation
from .aligned_io import AlignedReader, IO_WORKSPACE_BYTES

FLAG = "DS41_NATIVE_METADATA"
_IO_BYTES = 8 * 1024**2
_HEADER_BYTES = 1024**2
_TYPES = {"BF16": ("uint16", 2), "U32": ("uint32", 4),
          "F32": ("float32", 4), "F16": ("float16", 2),
          "U8": ("uint8", 1), "I8": ("int8", 1)}


def enabled_proof():
    value = os.environ.get(FLAG)
    if value in (None, "", "0"):
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{FLAG} requires an absolute proof path")
    return path


def _sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _identity(stat):
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _header(path):
    """Read only a bounded safetensors header, never a model payload."""
    with Path(path).open("rb") as stream:
        stat = os.fstat(stream.fileno())
        raw = stream.read(8)
        if len(raw) != 8:
            raise ValueError("Truncated metadata header")
        length = struct.unpack("<Q", raw)[0]
        if not 2 <= length <= min(_HEADER_BYTES, stat.st_size - 8):
            raise ValueError("Invalid metadata header extent")
        raw = stream.read(length)
        if len(raw) != length:
            raise ValueError("Truncated metadata header")
        header = json.loads(raw)
        if not isinstance(header, dict):
            raise ValueError("Invalid metadata header")
        if _identity(stat) != _identity(os.fstat(stream.fileno())):
            raise ValueError("Metadata file changed during header inspection")
        return header, 8 + length, _identity(stat)


def _entry(header, key, base, size, *, dtype, shape):
    entry = header.get(key)
    if not isinstance(entry, dict):
        raise ValueError(f"Missing metadata tensor: {key}")
    offsets = entry.get("data_offsets")
    if (entry.get("dtype") != dtype or entry.get("shape") != list(shape)
            or not isinstance(offsets, list) or len(offsets) != 2
            or any(type(v) is not int for v in offsets)):
        raise ValueError(f"Invalid metadata tensor: {key}")
    begin, end = offsets
    width = _TYPES[dtype][1]
    if begin < 0 or end - begin != math.prod(shape) * width or base + end > size:
        raise ValueError(f"Invalid metadata tensor range: {key}")
    return begin, end


@dataclass(frozen=True)
class MetadataRecord:
    name: str
    path: Path
    shape: tuple
    dictionary_values: int
    encoded_bytes: int
    original_bytes: int
    payload_sha256: str
    base: int
    identity: tuple
    index_range: tuple
    dictionary_range: tuple


_ARITHMETIC = r"""
template<int VPT>
inline float ds41_native_meta_load3(const device bfloat16_t* x, thread float* v) {
    float sum=0;
    for(int i=0;i<VPT;i+=8) {
        sum+=x[i]+x[i+1]+x[i+2]+x[i+3]+x[i+4]+x[i+5]+x[i+6]+x[i+7];
        v[i]=x[i];v[i+1]=x[i+1]/8.0f;v[i+2]=x[i+2]/64.0f;
        v[i+3]=x[i+3]/2.0f;v[i+4]=x[i+4]/16.0f;
        v[i+5]=x[i+5]/128.0f;v[i+6]=x[i+6]/4.0f;v[i+7]=x[i+7]/32.0f;
    }
    return sum;
}
template<int VPT>
inline float ds41_native_meta_dot3(const device uchar* w,thread float* v,float s,float b,float sum) {
    float a=0;
    for(int i=0;i<VPT/8;i++) {
        v+=8*i;w+=3*i;
        a+=(w[0]&0x07)*v[0];a+=(w[0]&0x38)*v[1];a+=(w[0]&0xc0)*v[2];
        a+=(w[1]&0x01)*(v[2]*256.0f);a+=(w[1]&0x0e)*v[3];a+=(w[1]&0x70)*v[4];
        a+=(w[1]&0x80)*v[5];a+=(w[2]&0x03)*(v[5]*256.0f);
        a+=(w[2]&0x1c)*v[6];a+=(w[2]&0xe0)*v[7];
    }
    return s*a+sum*b;
}
"""

_PROJECT = r"""
const uint lane=thread_index_in_simdgroup;
const uint row=threadgroup_position_in_grid.y*8+simdgroup_index_in_threadgroup*4;
const uint route=threadgroup_position_in_grid.x;
const uint expert=ids[route],source=left[route];
if(expert>=EXPERTS || source>=uint(x_shape[0])) {
    if(lane==0)for(uint r=0;r<4;r++)output[route*N+row+r]=bfloat16_t(0);
    return;
}
constexpr uint VPT=FAST ? 16 : 8;
constexpr uint BLOCK=VPT*32;
const uint mr=expert*N+row;
const device uchar* w=(const device uchar*)weight+size_t(mr)*(K*3/8)+lane*(VPT*3/8);
const device bfloat16_t* input=x+size_t(source)*K+lane*VPT;
uint meta=mr*(K/128)+lane/(128/VPT);
const device ushort* indices=(const device ushort*)joint_words;
float v[VPT],result[4]={0,0,0,0};
for(uint k=0;k<K;k+=BLOCK) {
    const float sum=ds41_native_meta_load3<VPT>(input,v);
    for(uint r=0;r<4;r++) {
        const uint code=indices[meta+r*(K/128)];
        const uint pair=code<uint(joint_dictionary_shape[0]) ? joint_dictionary[code] : 0;
        const float s=as_type<bfloat16_t>(ushort(pair&65535));
        const float b=as_type<bfloat16_t>(ushort(pair>>16));
        result[r]+=ds41_native_meta_dot3<VPT>(w+r*(K*3/8),v,s,b,sum);
    }
    w+=BLOCK*3/8;input+=BLOCK;meta+=BLOCK/128;
}
for(uint r=0;r<4;r++) {
    const float total=simd_sum(result[r]);
    if(lane==0)output[route*N+row+r]=bfloat16_t(total);
}
"""


@cache
def _project_kernel():
    return mx.fast.metal_kernel(
        name="ds41_native_joint_u16_g128_qmv3",
        input_names=["x", "weight", "joint_words", "joint_dictionary", "left", "ids"],
        output_names=["output"], header=_ARITHMETIC, source=_PROJECT)


@cache
def _decode_kernel():
    return mx.fast.metal_kernel(
        name="ds41_native_joint_u16_g128_decode",
        input_names=["joint_words", "joint_dictionary"],
        output_names=["scales", "biases"], source=r"""
const uint i=thread_position_in_grid.x;
if(size_t(i)<size_t(joint_words_shape[0])*2) {
    const uint code=((const device ushort*)joint_words)[i];
    const uint pair=code<uint(joint_dictionary_shape[0]) ? joint_dictionary[code] : 0;
    scales[i]=as_type<bfloat16_t>(ushort(pair&65535));
    biases[i]=as_type<bfloat16_t>(ushort(pair>>16));
}
""")


class JointMetadataProjection(QuantizedProjection):
    """Only packed weights, U16 codes and the raw BF16 pair table are resident."""

    def __init__(self, weight, words, dictionary, shape, **spec):
        nn.Module.__init__(self)
        shape = tuple(shape)
        if (spec.get("bits") != 3 or spec.get("mode") != "affine"
                or spec.get("group_size") != 128
                or weight.dtype != mx.uint32 or weight.ndim != 3
                or words.dtype != mx.uint32 or words.ndim != 1
                or dictionary.dtype != mx.uint32 or dictionary.ndim != 1
                or not 1 <= dictionary.size <= 65536 or len(shape) != 3
                or any(type(d) is not int or d <= 0 for d in shape)
                or math.prod(shape) % 2 or words.size != math.prod(shape) // 2
                or shape != (*weight.shape[:-1], weight.shape[-1] * 32 // 3 // 128)
                or weight.shape[-1] * 32 % (3 * 128)):
            raise ValueError("Invalid native affine3/G128 metadata projection")
        self.weight = weight
        self.joint_words, self.joint_dictionary = words, dictionary
        self._metadata_shape = shape
        self.bits, self.mode, self.group_size = 3, "affine", 128
        self.quantize_input = spec.get("quantize_input", True)
        if type(self.quantize_input) is not bool:
            raise ValueError("Invalid native metadata activation policy")

    @property
    def input_dims(self):
        return self.weight.shape[-1] * 32 // 3

    def decode_metadata(self):
        count = math.prod(self._metadata_shape)
        if mx.default_device() == mx.cpu:
            # Bit operations preserve signed zero, NaN payloads and all BF16 words.
            codes = self.joint_words.view(mx.uint16).astype(mx.uint32)
            pairs = self.joint_dictionary[codes].reshape(self._metadata_shape)
            return ((pairs & 65535).astype(mx.uint16).view(mx.bfloat16),
                    (pairs >> 16).astype(mx.uint16).view(mx.bfloat16))
        return _decode_kernel()(
            inputs=[self.joint_words, self.joint_dictionary],
            grid=(count, 1, 1), threadgroup=(256, 1, 1),
            output_shapes=[self._metadata_shape, self._metadata_shape],
            output_dtypes=[mx.bfloat16, mx.bfloat16])

    def dense(self):
        scales, biases = self.decode_metadata()
        return QuantizedProjection(self.weight, scales, 3, "affine", biases=biases,
                                   group_size=128, quantize_input=self.quantize_input)

    def project_quantized(self, x, indices=None, sorted_indices=False, block_plan=None):
        # Direct calls outside compact_expert retain ordinary MLX arithmetic.
        output = self.dense().project_quantized(x, indices, sorted_indices, block_plan)
        mx.eval(output)
        return output


def project(x, projection, left, ids):
    """Original MLX affine3 QMV with fixed-U16 pair lookup; short blocks only."""
    if not isinstance(projection, JointMetadataProjection):
        raise ValueError("A native G128 metadata projection is required")
    experts, n, _ = projection.weight.shape
    k = projection.input_dims
    if (mx.default_device() != mx.gpu or x.dtype != mx.bfloat16
            or k % 256 or n % 8 or not x.size or x.size % k
            or left.dtype != mx.uint32 or ids.dtype != mx.uint32
            or left.ndim != 1 or ids.ndim != 1 or not ids.size
            or left.size != ids.size):
        raise ValueError("Unsupported native metadata QMV geometry")
    return _project_kernel()(
        inputs=[x.reshape(-1, 1, k), projection.weight, projection.joint_words,
                projection.joint_dictionary, left, ids],
        template=[("K", k), ("N", n), ("EXPERTS", experts), ("FAST", k % 512 == 0)],
        grid=(32 * ids.size, n // 8 * 2, 1), threadgroup=(32, 2, 1),
        output_shapes=[(ids.size, 1, n)], output_dtypes=[mx.bfloat16])[0]


def compact_expert(original, expert, x, indices=None, weights=None,
                   sorted_indices=False, *, input_quantized=False,
                   max_grouped_tokens=8):
    """Use direct compact QMV for decode; release one dense bank per prefill."""
    projections = (expert.w1, expert.w3, expert.w2)
    if not any(isinstance(p, JointMetadataProjection) for p in projections):
        return original(expert, x, indices, weights, sorted_indices,
                        input_quantized=input_quantized,
                        max_grouped_tokens=max_grouped_tokens)
    if not all(isinstance(p, JointMetadataProjection) for p in projections):
        raise ValueError("Partially compacted expert bank is unsupported")
    if input_quantized and not expert.quantizes_input:
        raise ValueError("Prequantized input requires matching quantized projections")
    k = expert.w1.input_dims
    rows = x.size // k
    if (indices is not None and weights is not None and not sorted_indices
            and x.dtype == mx.bfloat16 and mx.default_device() == mx.gpu
            and all(p.quantize_input for p in projections)
            # Ragged target verification explicitly permits up to32 token
            # rows. Each route keeps the same independent QMV reduction;
            # default decode calls retain their original eight-row bound.
            and 1 <= rows <= min(32, max_grouped_tokens)
            and x.size % k == 0 and indices.size > 0
            and indices.shape[-1] > 0
            and indices.size // indices.shape[-1] == rows
            and weights.size == indices.size
            and all(p.input_dims % 256 == 0 and p.output_dims % 8 == 0
                    for p in projections)):
        from .activation import quantize_swiglu_activation
        inp = (x if input_quantized else quantize_activation(x)).reshape(-1, 1, k)
        ids = indices.reshape(-1).astype(mx.uint32)
        left = mx.arange(ids.size, dtype=mx.uint32) // indices.shape[-1]
        gate = project(inp, expert.w1, left, ids)
        up = project(inp, expert.w3, left, ids)
        y = quantize_swiglu_activation(gate, up, weights.reshape(-1), x.dtype, expert._limit)
        down = project(y, expert.w2, mx.arange(ids.size, dtype=mx.uint32), ids)
        return down.reshape(*indices.shape, 1, x.shape[-1])
    return _dense_expert(original, expert, x, indices, weights, sorted_indices,
                         input_quantized=input_quantized,
                         max_grouped_tokens=max_grouped_tokens)


def _dense_expert(original, expert, x, indices, weights, sorted_indices,
                  *, input_quantized, max_grouped_tokens):
    # Do not keep per-layer dense metadata in nn.Module.parameters or the graph.
    projections = (expert.w1, expert.w3, expert.w2)
    dense = expert.__class__.__new__(expert.__class__)
    nn.Module.__init__(dense)
    dense.w1, dense.w3, dense.w2 = (p.dense() for p in projections)
    dense._limit = expert._limit
    output = original(dense, x, indices, weights, sorted_indices,
                      input_quantized=input_quantized,
                      max_grouped_tokens=max_grouped_tokens)
    mx.eval(output)
    del dense
    return output


def compact_moe(original, moe, x, image_mask):
    """Retain oMLX's source-row NAX prefill without routed-row duplication."""
    projections = (moe.experts.w1, moe.experts.w3, moe.experts.w2)
    if not any(isinstance(p, JointMetadataProjection) for p in projections):
        return original(moe, x, image_mask)
    if not all(isinstance(p, JointMetadataProjection) for p in projections):
        raise ValueError("Partially compacted expert bank is unsupported")
    from . import affine_gather
    from .language import DS41_GATHER
    if not (DS41_GATHER and affine_gather.eligible(moe, x)):
        # Short decode continues through the unchanged MoE gate/combine, then
        # the compact_expert branch. Other prefill geometries retain the
        # ordinary expert path's sorted/unsorted decisions.
        return original(moe, x, image_mask)
    from . import native_metadata_prefill
    if native_metadata_prefill.eligible(moe, x):
        return native_metadata_prefill.forward(moe, x, image_mask)
    return _dense_prefill_moe(original, moe, x, image_mask)


def _dense_prefill_moe(original, moe, x, image_mask):
    """Qualified bulk fallback; charge its transient even with direct enabled."""
    projections = (moe.experts.w1, moe.experts.w3, moe.experts.w2)
    dense_expert = moe.experts.__class__.__new__(moe.experts.__class__)
    nn.Module.__init__(dense_expert)
    dense_expert.w1, dense_expert.w3, dense_expert.w2 = (p.dense() for p in projections)
    dense_expert._limit = moe.experts._limit
    dense_moe = moe.__class__.__new__(moe.__class__)
    nn.Module.__init__(dense_moe)
    dense_moe.gate, dense_moe.experts, dense_moe.shared_experts = moe.gate, dense_expert, moe.shared_experts
    dense_moe._a8_min_tokens = getattr(moe, "_a8_min_tokens", None)
    output = original(dense_moe, x, image_mask)
    mx.eval(output)
    del dense_moe, dense_expert
    return output


class NativeMetadataPlan:
    """Header-only admission plus bounded checked loading of existing files."""

    def __init__(self, checkpoint, proof_path, *, allocation_check=None):
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        self.proof_path = Path(proof_path).expanduser().resolve()
        self.root = self.proof_path.parent
        self.allocation_check = allocation_check
        proof = json.loads(self.proof_path.read_text())
        self.proof_sha256 = _sha(self.proof_path)
        sidecar_path = self.root / "DeepSeek-V4.1-LSQ3-G128-exact-metadata.json"
        sidecar = json.loads(sidecar_path.read_text())
        if (proof.get("status") != "passed" or proof.get("quantization_codes_changed") is not False
                or proof.get("original_tensors_retained") is not True
                or sidecar.get("requantized") is not False
                or sidecar.get("source_index_sha256") != _sha(self.checkpoint / "model.safetensors.index.json")
                or sidecar.get("source_config_sha256") != _sha(self.checkpoint / "config.json")):
            raise ValueError("Native metadata checkpoint identity mismatch")
        # The native sidecar pins every original source header, including MTP,
        # vision and SSD Engram. Header-only validation retains that identity
        # without rereading hundreds of GiB of original payloads.
        self.source_identities = {}
        self.encoded_identities = {}
        encoded_names = {record.get("file") for record in proof.get("projections", [])}
        pins = sidecar.get("files")
        if not isinstance(pins, list) or not pins:
            raise ValueError("Missing native source header identities")
        for pin in pins:
            source_path = (self.root / pin["path"]).resolve()
            original = source_path.is_relative_to(self.checkpoint)
            encoded = source_path.parent == self.root and source_path.name in encoded_names
            if not original and not encoded:
                raise ValueError("Native pinned source escapes checkpoint")
            with source_path.open("rb") as stream:
                stat = os.fstat(stream.fileno())
                raw = stream.read(8)
                if len(raw) != 8:
                    raise ValueError("Truncated native pinned source header")
                length = struct.unpack("<Q", raw)[0]
                if not 2 <= length <= min(_HEADER_BYTES, stat.st_size - 8):
                    raise ValueError("Invalid native pinned source header extent")
                raw += stream.read(length)
                if (len(raw) != length + 8 or stat.st_size != pin.get("size")
                        or stat.st_mtime_ns != pin.get("mtime_ns")
                        or len(raw) != pin.get("header_bytes")
                        or hashlib.sha256(raw).hexdigest() != pin.get("header_sha256")
                        or _identity(stat) != _identity(os.fstat(stream.fileno()))):
                    raise ValueError("Native pinned source header identity mismatch")
                identities = self.source_identities if original else self.encoded_identities
                if source_path in identities:
                    raise ValueError("Duplicate native pinned file identity")
                identities[source_path] = _identity(stat)
        raw = json.loads((self.checkpoint / "config.json").read_text())
        format_spec = raw.get("omlx_deepseek_v41") or {}
        if format_spec.get("version") != 1 or format_spec.get("preserve_mtp") is not True:
            raise ValueError("Native metadata requires a preserved converted checkpoint")
        mapping = json.loads((self.checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
        quantized = format_spec.get("quantized_modules") or {}
        prefixes = {key.rsplit(".", 1)[0] for key in mapping
                    if key.startswith("language_model.layers.") and ".ffn.experts." in key
                    and key.endswith((".scales", ".biases"))}
        expected = {f"language_model.layers.{i}.ffn.experts.{projection}"
                    for i in range(40) for projection in ("w1", "w3", "w2")}
        if prefixes != expected:
            raise ValueError("Native metadata requires the complete 40-layer routed bank")
        records = proof.get("projections")
        if not isinstance(records, list) or len(records) != len(expected):
            raise ValueError("Incomplete native metadata proof")
        by_name = {record.get("prefix"): record for record in records}
        if len(by_name) != len(expected) or set(by_name) != expected:
            raise ValueError("Duplicate or missing native metadata proof projection")
        headers = {}
        self.records = {}
        for name in sorted(expected):
            record, spec = by_name[name], quantized.get(name) or {}
            if (spec.get("bits") != 3 or spec.get("group_size") != 128
                    or spec.get("mode") != "affine" or spec.get("quantize_input") is not True
                    or record.get("encoded") is not True
                    or record.get("all_words_roundtrip_exact") is not True
                    or record.get("full_encoded_readback_passed") is not True):
                raise ValueError(f"Unsupported or unverified native metadata: {name}")
            filename = record.get("file")
            if not isinstance(filename, str) or Path(filename).name != filename:
                raise ValueError("Native metadata path escapes its directory")
            path = (self.root / filename).resolve()
            if path.parent != self.root or not path.is_file():
                raise ValueError("Native metadata file is missing or escapes its directory")
            source = []
            for field in ("weight", "scales", "biases"):
                key = name + "." + field
                source_path = (self.checkpoint / mapping[key]).resolve()
                if not source_path.is_relative_to(self.checkpoint):
                    raise ValueError("Native source tensor path escapes checkpoint")
                if source_path not in headers:
                    headers[source_path] = _header(source_path)
                header, base, identity = headers[source_path]
                if self.source_identities.get(source_path) != identity:
                    raise ValueError("Native metadata source is not in the pinned checkpoint")
                entry = header.get(key)
                if not isinstance(entry, dict):
                    raise ValueError("Missing native source metadata")
                shape = tuple(entry.get("shape", ()))
                if len(shape) != 3 or any(type(d) is not int or d <= 0 for d in shape):
                    raise ValueError("Invalid native source metadata shape")
                _entry(header, key, base, identity[2], dtype="U32" if field == "weight" else "BF16", shape=shape)
                source.append(shape)
            weight_shape, shape, bias_shape = source
            if (shape != bias_shape or shape != (*weight_shape[:-1], weight_shape[-1] * 32 // 3 // 128)
                    or weight_shape[-1] * 32 % (3 * 128) or math.prod(shape) % 2):
                raise ValueError("Native source metadata geometry mismatch")
            header, base, identity = _header(path)
            if self.encoded_identities.get(path) != identity:
                raise ValueError("Native encoded header is not in the pinned package")
            dictionary_values = record.get("dictionary_values")
            if type(dictionary_values) is not int or not 1 <= dictionary_values <= 65536:
                raise ValueError("Invalid native metadata dictionary size")
            if set(header) - {"__metadata__"} != {"indices", "dictionary"}:
                raise ValueError("Unexpected native metadata file tensor")
            index_range = _entry(header, "indices", base, identity[2], dtype="U32", shape=(math.prod(shape) // 2,))
            dictionary_range = _entry(header, "dictionary", base, identity[2], dtype="U32", shape=(dictionary_values,))
            encoded = math.prod(shape) * 2 + dictionary_values * 4
            original = math.prod(shape) * 4
            if (index_range != (0, math.prod(shape) * 2)
                    or dictionary_range != (index_range[1], encoded)
                    or base + encoded != identity[2]
                    or record.get("original_bytes") != original
                    or record.get("encoded_bytes") != encoded
                    or record.get("saved_bytes") != original - encoded
                    or not isinstance(record.get("payload_sha256"), str)
                    or len(record["payload_sha256"]) != 64):
                raise ValueError("Native metadata size or proof mismatch")
            self.records[name] = MetadataRecord(name, path, shape, dictionary_values,
                encoded, original, record["payload_sha256"], base, identity, index_range, dictionary_range)
        self.keys = frozenset(name + "." + field for name in self.records for field in ("scales", "biases"))
        self._loaded = set()
        self.saved_bytes = sum(record.original_bytes - record.encoded_bytes for record in self.records.values())
        self.original_bytes = sum(record.original_bytes for record in self.records.values())
        self.encoded_bytes = sum(record.encoded_bytes for record in self.records.values())
        self.prefill_transient_bytes = max(sum(self.records[f"language_model.layers.{layer}.ffn.experts.{p}"].original_bytes
                                         for p in ("w1", "w3", "w2")) for layer in range(40))
        if (proof.get("exact_dictionary_projections") != len(self.records)
                or proof.get("saved_resident_bytes") != self.saved_bytes):
            raise ValueError("Native metadata proof totals mismatch")
        self.excluded_mtp_keys = frozenset()
        self.excluded_mtp_bytes = 0
        self.draft_offload_plan = None
        self.woa_plan = None

    def contains(self, name):
        return name in self.records

    def _check(self, requested, stage):
        if self.allocation_check is not None:
            self.allocation_check(requested, stage=stage)

    def read_original(self, reader, key):
        if key in self.keys:
            raise ValueError("Original native metadata must not be loaded")
        item = reader.header[key]
        stream_path = Path(reader._file.name).resolve()
        expected_identity = self.source_identities.get(stream_path)
        if expected_identity is None or _identity(os.fstat(reader._file.fileno())) != expected_identity:
            raise ValueError("Native original source changed before reading")
        dtype = item["dtype"]
        if dtype not in _TYPES:
            raise ValueError(f"Unsupported native direct-load dtype: {dtype}")
        name, width = _TYPES[dtype]
        start, end = item["data_offsets"]
        if end - start != math.prod(item["shape"]) * width:
            raise ValueError("Invalid native direct-load tensor extent")
        self._check(end - start, "allocate original " + key)
        value = mx.contiguous(mx.zeros(item["shape"], dtype=getattr(mx, name)))
        mx.eval(value)
        self._read_into(reader._file, reader._start + start, value, None, key)
        if _identity(os.fstat(reader._file.fileno())) != expected_identity:
            raise ValueError("Native original source changed during reading")
        return value.view(mx.bfloat16) if dtype == "BF16" else value

    def _read_into(self, stream, offset, value, digest, stage):
        """Fill evaluated unified storage; at most one bounded read is pending."""
        if value.size:
            # Establish GPU residency before exposing the shared CPU pointer.
            # Explicit CPU fixtures skip this GPU submission entirely.
            if mx.default_device() == mx.gpu:
                mx.eval(mx.sum(value.reshape(-1)[:1].astype(mx.float32)))
                mx.synchronize()
            host = np.asarray(value)
            target = memoryview(host).cast("B")
            self._check(IO_WORKSPACE_BYTES, "aligned I/O workspace " + stage)
            with AlignedReader.from_fd(stream.fileno(), closefd=False) as reader:
                reader.seek(offset)
                count = 0
                while count < len(target):
                    end = min(count + _IO_BYTES, len(target))
                    read = reader.readinto(target[count:end])
                    if not read:
                        raise ValueError("Truncated native metadata/weight payload")
                    if digest is not None:
                        digest.update(target[count:count + read])
                    count += read
                    self._check(0, "read " + stage)
            del target, host

    def _read(self, name):
        record = self.records[name]
        self._check(record.encoded_bytes, "allocate native metadata " + name)
        words = mx.contiguous(mx.zeros((math.prod(record.shape) // 2,), dtype=mx.uint32))
        dictionary = mx.contiguous(mx.zeros((record.dictionary_values,), dtype=mx.uint32))
        mx.eval(words, dictionary)
        digest = hashlib.sha256()
        with record.path.open("rb") as stream:
            if _identity(os.fstat(stream.fileno())) != record.identity:
                raise ValueError("Native metadata file changed before loading")
            self._read_into(stream, record.base + record.index_range[0], words, digest, name + " indices")
            self._read_into(stream, record.base + record.dictionary_range[0], dictionary, digest, name + " dictionary")
            if digest.hexdigest() != record.payload_sha256:
                raise ValueError("Native metadata payload checksum mismatch")
            if _identity(os.fstat(stream.fileno())) != record.identity:
                raise ValueError("Native metadata file changed during loading")
        # Content checks are integer-only and bounded to one code bank; no BF16
        # source values or dense metadata are created. All native files were also
        # completely round-trip checked against the original source words.
        host = np.asarray(dictionary)
        if host.size > 1 and not np.all(host[1:] > host[:-1]):
            raise ValueError("Native metadata dictionary must be sorted and unique")
        for begin in range(0, words.size, _IO_BYTES // 4):
            chunk = np.asarray(words[begin:begin + _IO_BYTES // 4]).view(np.uint16)
            if chunk.size and int(chunk.max()) >= record.dictionary_values:
                raise ValueError("Native metadata dictionary index out of bounds")
        self._loaded.update((name + ".scales", name + ".biases"))
        return words, dictionary

    def projection(self, name, weight, spec, logical):
        record = self.records[name]
        shape = (*logical[:-1], logical[-1] // 128)
        if (record.shape != shape or weight.shape != (*logical[:-1], logical[-1] * 3 // 32)
                or spec.get("bits") != 3 or spec.get("group_size") != 128
                or spec.get("mode") != "affine"):
            raise ValueError("Invalid native metadata projection geometry")
        words, dictionary = self._read(name)
        return JointMetadataProjection(weight, words, dictionary, shape, **spec)

    def original_parameter_keys(self, keys):
        result = set(keys)
        for name in self.records:
            compressed = {name + ".joint_words", name + ".joint_dictionary"}
            if not compressed.issubset(result):
                raise ValueError("Missing resident native metadata parameter")
            result.difference_update(compressed)
        result.update(self.keys)
        return result

    def validate_complete(self):
        if self._loaded != self.keys:
            raise ValueError("Not all native metadata projections were loaded")
        if _sha(self.proof_path) != self.proof_sha256:
            raise ValueError("Native metadata proof changed during loading")
        for path, identity in self.source_identities.items():
            if _identity(path.stat()) != identity:
                raise ValueError("Native source checkpoint changed during loading")

    def residency_estimate(self, estimate):
        """Preserve admission margins while replacing only physical metadata."""
        if not estimate.supported:
            raise ValueError("Native metadata checkpoint residency is unsupported")
        # The ordinary header estimator includes 5% and 32 MiB. Retain both;
        # independent transient accounting must additionally charge one dense
        # bank during prefill. No blanket process/free-memory policy is changed.
        saved_with_existing_margin = int(self.saved_bytes * 1.05)
        if not 0 < saved_with_existing_margin < estimate.mmap_bytes:
            raise ValueError("Invalid native metadata residency saving")
        return replace(estimate, mmap_bytes=estimate.mmap_bytes - saved_with_existing_margin,
                       resident_bytes=estimate.resident_bytes - saved_with_existing_margin)

    def summary(self):
        return {"format": "fixed-u16-bf16-pair-g128", "projections": len(self.records),
                "original_metadata_bytes": self.original_bytes,
                "resident_metadata_bytes": self.encoded_bytes,
                "saved_bytes": self.saved_bytes,
                "prefill_dense_bank_bytes": self.prefill_transient_bytes,
                "draft_tensors_excluded": 0, "requantized": False,
                "proof_sha256": self.proof_sha256}
