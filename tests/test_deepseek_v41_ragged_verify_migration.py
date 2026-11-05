"""Ragged target verification retains request-local results and shared weights."""
import mlx.core as mx
import numpy as np
import pytest
from test_deepseek_v41 import load_reference_weights
from test_deepseek_v41_mtp import mtp_config
from omlx.patches.deepseek_v41.language import LanguageModel


def packed_lsq_model(compact=True, depth=3):
    from dataclasses import replace
    import mlx.nn as nn
    from mlx.utils import tree_flatten, tree_unflatten
    from omlx.patches.deepseek_v41.native_metadata import JointMetadataProjection
    from omlx.patches.deepseek_v41.quantization import QuantizedProjection
    config = mtp_config(dim=256, moe_inter_dim=256, q_lora_rank=256,
        n_heads=2, head_dim=128, o_lora_rank=128, max_seq_len=256, max_batch_size=8)
    if depth == 4:
        config = replace(config, dspark_block_size=5)
    model = LanguageModel(config)
    load_reference_weights(model, scale=0.03)
    model.update(tree_unflatten([(name, value.astype(mx.bfloat16)) for name, value in tree_flatten(model.parameters())
                  if name.endswith("weight") and ".gate." not in name]))
    replacements = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and not name.endswith(("wo_a", "gate", "head")):
            weight, scales = mx.quantize(module.weight, group_size=32, bits=8, mode="mxfp8")
            replacements.append((name, QuantizedProjection(weight, scales, 8, "mxfp8")))
    model.update_modules(tree_unflatten(replacements))
    for layer in model.layers:
        expert = layer.ffn.experts
        for name in ("w1", "w3", "w2"):
            weight, scales, biases = mx.quantize(getattr(expert, name).weight,
                group_size=128, bits=3, mode="affine")
            if compact:
                pairs = np.asarray(scales.view(mx.uint16)).astype(np.uint32)
                pairs |= np.asarray(biases.view(mx.uint16)).astype(np.uint32) << 16
                dictionary, codes = np.unique(pairs, return_inverse=True)
                projection = JointMetadataProjection(weight, mx.array(codes.astype(np.uint16).ravel().view(np.uint32)),
                    mx.array(dictionary), scales.shape, bits=3, mode="affine", group_size=128,
                    quantize_input=True)
            else:
                projection = QuantizedProjection(weight, scales, 3, "affine", biases=biases,
                    group_size=128, quantize_input=True)
            setattr(expert, name, projection)
        expert._native_metadata = compact
    weight, scales, biases = mx.quantize(model.head.weight, group_size=64, bits=8)
    model.head = QuantizedProjection(weight, scales, 8, "affine", biases=biases,
                                     group_size=64, quantize_input=False)
    mx.eval(model.parameters())
    model.configure_mtp(True, depth)
    return model


@pytest.mark.parametrize("lengths", [(2, 5), (1, 2, 3, 4, 5, 1, 2, 3)])
def test_ragged_target_forward_matches_request_local_reference(lengths):
    model = LanguageModel(mtp_config(max_seq_len=256, max_batch_size=8))
    load_reference_weights(model)
    model.configure_mtp(True, 3)
    actual, expected = [], []
    inputs=[]
    for i, length in enumerate(lengths):
        a,b=model.make_cache(),model.make_cache()
        prompt=mx.array([[3+j%10 for j in range(3+i)]])
        model(prompt,cache=a,return_hidden=True)
        model(prompt,cache=b,return_hidden=True)
        actual.append(a);expected.append(b)
        inputs.append(mx.array([[20+j for j in range(length)]]))
    wanted=[model(ids,cache=cache,return_hidden=True,target_verify=True) for ids,cache in zip(inputs,expected)]
    results=model.mtp_verify_requests(inputs,actual)
    for (logits,hidden,_),(ref_logits,ref_hidden),a,b in zip(results,wanted,actual,expected):
        np.testing.assert_array_equal(logits.view(mx.uint32),ref_logits.view(mx.uint32))
        np.testing.assert_array_equal(hidden.view(mx.uint32),ref_hidden.view(mx.uint32))
        for ac,bc in zip(a,b):
            assert ac.size()==bc.size()
            for x,y in zip(ac.cache,bc.cache):
                if x is not None:np.testing.assert_array_equal(x.view(mx.uint8),y.view(mx.uint8))


@pytest.mark.parametrize("lengths", [(2, 5), (1, 2, 3, 4, 5, 1, 2, 3), (4,)*8, (5,)*8])
@pytest.mark.parametrize("compact", [False, True])
def test_lsq_ragged_target_matches_serial_and_preserves_rollback(lengths, compact, monkeypatch):
    from omlx.patches.deepseek_v41 import batched_verify, native_metadata
    model = packed_lsq_model(compact)
    actual, expected, inputs = [], [], []
    for i, length in enumerate(lengths):
        a, b = model.make_cache(), model.make_cache()
        prompt = mx.array([[3+j%10 for j in range(3+i)]])
        for cache in (a, b):
            model(prompt, cache=cache, return_hidden=True)
        actual.append(a); expected.append(b)
        inputs.append(mx.array([[20+j for j in range(length)]]))
    wanted = [model(ids, cache=cache, return_hidden=True, target_verify=True)
              for ids, cache in zip(inputs, expected)]
    mx.eval(wanted)
    calls = []
    original = batched_verify.routed_forward
    def observed(moe, values):
        calls.append(tuple(value.shape[1] for value in values))
        return original(moe, values)
    monkeypatch.setattr(batched_verify, "routed_forward", observed)
    if compact:
        monkeypatch.setattr(native_metadata, "_dense_expert", lambda *a, **k:
            (_ for _ in ()).throw(AssertionError("Compact short blocks must stay native")))
    results = model.mtp_verify_requests(inputs, actual)
    cohorts, current, count = [], [], 0
    for length in lengths:
        if count + length > 32:
            cohorts.append(tuple(current)); current, count = [], 0
        current.append(length); count += length
    cohorts.append(tuple(current))
    assert calls == [group for cohort in cohorts for group in [cohort]*len(model.layers)]
    for index, ((logits, hidden, _), (ref_logits, ref_hidden)) in enumerate(zip(results, wanted)):
        np.testing.assert_array_equal(logits.view(mx.uint32), ref_logits.view(mx.uint32))
        np.testing.assert_array_equal(hidden.view(mx.uint16), ref_hidden.view(mx.uint16))
        accepted = min(1, lengths[index]-1)
        assert model.mtp_partial_rollback(actual[index], accepted, lengths[index]-1)
        assert model.mtp_partial_rollback(expected[index], accepted, lengths[index]-1)
        assert actual[index][0]._mtp_draft_stash is None
        for a, b in zip(actual[index], expected[index]):
            assert a.size() == b.size() == 3+index+accepted+1
            assert getattr(a, "_mtp_verify_state", None) is None
            for x, y in zip(a.cache, b.cache):
                if x is not None: np.testing.assert_array_equal(x.view(mx.uint8), y.view(mx.uint8))


@pytest.mark.parametrize("requests", [2, 8])
@pytest.mark.parametrize("depth", [3, 4])
def test_compact_lsq_complete_generation_matches_serial_and_cleans_up(requests, depth, monkeypatch):
    from mlx_lm.generate import BatchGenerator
    from omlx.patches.mlx_lm_mtp import batch_generator
    from omlx.patches.deepseek_v41 import batched_verify, native_metadata
    batch_generator.apply()
    model = packed_lsq_model(True, depth=depth)
    prompts = [[3+j%10 for j in range(3+i)] for i in range(requests)]
    expected = []
    model.configure_mtp(False, 3)
    for prompt in prompts:
        cache, result = model.make_cache(), []
        # Fix the prefill program before comparing serial/concurrent decode,
        # as the production scheduler does for preserved CED chunks.
        model(mx.array([prompt[:-1]]), cache=cache)
        ids = mx.array([[prompt[-1]]])
        for _ in range(12):
            token = int(mx.argmax(model(ids, cache=cache)[0, -1]).item())
            result.append(token); ids = mx.array([[token]])
        expected.append(result)
    model.configure_mtp(True, depth)
    cached_prompts = []
    for prompt in prompts:
        cache = model.make_cache()
        model(mx.array([prompt[:-1]]), cache=cache)
        cached_prompts.append(cache)
    target_calls = []
    original = batched_verify.forward
    def observed(host, inputs, caches):
        target_calls.append(tuple(ids.shape[1] for ids in inputs))
        return original(host, inputs, caches)
    monkeypatch.setattr(batched_verify, "forward", observed)
    original_dense = native_metadata._dense_expert
    def no_decode_expansion(original, expert, x, *args, **kwargs):
        rows = x.size // expert.w1.input_dims
        assert rows > 8 and kwargs.get("max_grouped_tokens", 8) == 8, "Decode must retain compact native metadata"
        return original_dense(original, expert, x, *args, **kwargs)
    monkeypatch.setattr(native_metadata, "_dense_expert", no_decode_expansion)
    generator = BatchGenerator(model, max_tokens=12, completion_batch_size=8,
        prefill_batch_size=8, prefill_step_size=256, sampler=lambda logits: mx.argmax(logits, -1))
    result = {}
    try:
        uids = generator.insert([[prompt[-1]] for prompt in prompts],
            caches=cached_prompts, all_tokens=[prompt[:-1] for prompt in prompts])
        result = {uid: [] for uid in uids}
        for _ in range(128):
            prefill, responses = generator.next()
            for response in responses: result[response.uid].append(response.token)
            if all(len(tokens) == 12 for tokens in result.values()): break
        assert [result[uid] for uid in uids] == expected
        assert target_calls and all(sum(shape) <= 32 for shape in target_calls)
    finally:
        generator.remove(list(result))
        generator.close()
    assert not generator._generation_batch.uids if generator._generation_batch is not None else True
