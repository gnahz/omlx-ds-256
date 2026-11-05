"""Optional exact verification keeps the unchanged serial-token oracle.

Run only in the detached GPU qualification worker; the production adapter does
not select this method. Tiny LSQ3/G128 banks stay within the component limit.
"""
import mlx.core as mx
import numpy as np
import pytest
from test_deepseek_v41_ragged_verify_migration import packed_lsq_model


def assert_bits(actual,expected):
    assert actual.dtype==expected.dtype and actual.shape==expected.shape
    np.testing.assert_array_equal(np.asarray(actual.view(mx.uint8)),np.asarray(expected.view(mx.uint8)))


@pytest.mark.parametrize('length,accepted',[(2,0),(4,1),(5,0),(5,3),(5,4)])
def test_exact_verify_logits_hidden_and_rollback_match_single_tokens(length,accepted,monkeypatch):
    model=packed_lsq_model(True,depth=4)
    actual,expected=model.make_cache(),model.make_cache()
    for cache in (actual,expected): model(mx.array([[3,4,5]]),cache=cache)
    ids=mx.array([[11+i for i in range(length)]])
    wanted=[model(ids[:,i:i+1],cache=expected,return_hidden=True) for i in range(length)]
    logits,hidden,_=model.mtp_verify_requests_exact([ids],[actual])[0]
    assert_bits(logits,mx.concatenate([v[0] for v in wanted],1))
    assert_bits(hidden,mx.concatenate([v[1] for v in wanted],1))
    with monkeypatch.context() as patch:
        patch.setattr(model,'_forward',lambda *a,**k: (_ for _ in ()).throw(AssertionError('Rollback must not execute the target')))
        assert model.mtp_partial_rollback(actual,accepted,length-1)
    reference=model.make_cache()
    model(mx.array([[3,4,5]]),cache=reference)
    for i in range(accepted+1): model(ids[:,i:i+1],cache=reference)
    for a,b in zip(actual,reference):
        assert a.size()==b.size()==4+accepted
        for x,y in zip(a.cache,b.cache):
            assert (x is None)==(y is None)
            if x is not None: assert_bits(x,y)
    assert actual[0]._mtp_draft_stash is None
    following=mx.array([[25]])
    assert_bits(model(following,cache=actual),model(following,cache=reference))


@pytest.mark.parametrize('requests,depth',[(2,3),(8,3),(2,4),(8,4)])
def test_optional_exact_generation_matches_unchanged_serial_tokens(requests,depth,monkeypatch):
    from mlx_lm.generate import BatchGenerator
    from omlx.patches.mlx_lm_mtp import batch_generator
    batch_generator.apply()
    model=packed_lsq_model(True,depth=depth)
    prompts=[[3+j%10 for j in range(3+i)] for i in range(requests)]
    expected=[]
    model.configure_mtp(False,3)
    for prompt in prompts:
        cache=model.make_cache();tokens=[]
        model(mx.array([prompt[:-1]]),cache=cache)
        ids=mx.array([[prompt[-1]]])
        for _ in range(12):
            token=int(mx.argmax(model(ids,cache=cache)[0,-1]).item())
            tokens.append(token);ids=mx.array([[token]])
        expected.append(tokens)
    model.configure_mtp(True,depth)
    caches=[]
    for prompt in prompts:
        cache=model.make_cache();model(mx.array([prompt[:-1]]),cache=cache);caches.append(cache)
    # Only this diagnostic test opts in; no runtime/default setting changes.
    calls=[]
    exact=model.mtp_verify_requests_exact
    def observed(inputs,states):
        calls.append(tuple(ids.shape[1] for ids in inputs))
        return exact(inputs,states)
    monkeypatch.setattr(model,'mtp_verify_requests',observed)
    generator=BatchGenerator(model,max_tokens=12,completion_batch_size=8,
        prefill_batch_size=8,prefill_step_size=256,sampler=lambda logits:mx.argmax(logits,-1))
    result={}
    try:
        uids=generator.insert([[prompt[-1]] for prompt in prompts],caches=caches,all_tokens=[prompt[:-1] for prompt in prompts])
        result={uid:[] for uid in uids}
        for _ in range(128):
            _,responses=generator.next()
            for response in responses:result[response.uid].append(response.token)
            if all(len(tokens)==12 for tokens in result.values()):break
        assert [result[uid] for uid in uids]==expected
        assert calls and all(1<=n<=8 for lengths in calls for n in lengths)
    finally:
        generator.remove(list(result));generator.close()
    assert not generator._generation_batch.uids if generator._generation_batch is not None else True


@pytest.mark.parametrize('requests',[2,8])
def test_optional_exact_small_fixture_latency(requests,record_property):
    """Diagnostic timing only; these tiny tensors do not predict model speed."""
    import statistics
    import time
    from omlx.patches.deepseek_v41.prefix_cache import clone_cache
    model=packed_lsq_model(True,depth=4)
    bases=[]
    inputs=[]
    for row in range(requests):
        cache=model.make_cache()
        model(mx.array([[3,4,5,6+row]]),cache=cache)
        bases.append(cache);inputs.append(mx.array([[11,12,13,14,15]]))
    timings={'block':[],'strict':[]}
    methods={'block':model.mtp_verify_requests,'strict':model.mtp_verify_requests_exact}
    for repeat in range(4):
        for name in ('block','strict') if repeat%2==0 else ('strict','block'):
            caches=[clone_cache(cache) for cache in bases]
            mx.synchronize()
            began=time.perf_counter()
            result=methods[name](inputs,caches)
            mx.eval([array for item in result for array in item if array is not None])
            mx.synchronize()
            elapsed=(time.perf_counter()-began)*1000
            if repeat:timings[name].append(elapsed)
    for name,samples in timings.items():
        record_property(name+'_median_ms',statistics.median(samples))
    record_property('strict_over_block_ratio',statistics.median(timings['strict'])/statistics.median(timings['block']))
    record_property('interpretation','Tiny synthetic dim256; excludes real model load and is not full-model throughput evidence')
