"""AST-only undo assembly checks; importing this file cannot initialize MLX."""
import ast
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

SOURCE = Path(__file__).resolve().parents[1] / 'omlx/patches/deepseek_v41/exact_verify.py'
namespace = {'mx': SimpleNamespace(concatenate=np.concatenate)}
tree=ast.parse(SOURCE.read_text())
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef)],type_ignores=[]),str(SOURCE),'exec'),namespace)
forward=namespace['forward']


class Cache:
    def __init__(self):
        self.offset=3
        self.cache=[np.array([3]),np.array([[[10],[11],[12]]]),None,None,np.array([[[20]]]),np.array([[[30]]])]
        self.left_padding=np.array([0]);self.lengths=np.array([3])
    def size(self): return self.offset


class Model:
    def __init__(self): self.calls=[]
    def __call__(self, ids, *, cache, return_hidden, target_verify):
        assert return_hidden and target_verify and ids.shape==(1,1)
        self.calls.append(int(ids[0,0]));states=[];before=cache[0].size()
        for item in cache:
            value=ids[...,None]
            state={'window':np.concatenate([item.cache[1],value],1),
                   'compressor':(np.concatenate([item.cache[4],value+20],1),np.concatenate([item.cache[5],value+30],1))}
            states.append(state)
            item.cache[1]=state['window'][:,-3:]
            n=state['compressor'][0].shape[1]//2*2
            item.cache[4],item.cache[5]=(part[:,n:] for part in state['compressor'])
            item.offset+=1
        cache[0]._mtp_draft_stash=(ids,None,before,states)
        return ids[...,None].astype(float), (ids[...,None]+100).astype(float)


def test_undo_keeps_initial_partial_group_and_each_new_row_once():
    model=Model();cache=[Cache(),Cache()];ids=np.array([[4,5,6,7]])
    output=forward(model,[ids],[cache])[0]
    assert model.calls==[4,5,6,7]
    np.testing.assert_array_equal(output[0],ids[...,None])
    np.testing.assert_array_equal(output[1],ids[...,None]+100)
    tokens,snapshots,before,states=cache[0]._mtp_draft_stash
    assert before==3 and cache[0].size()==7
    np.testing.assert_array_equal(tokens,ids)
    np.testing.assert_array_equal(snapshots[0][0][1],[[[10],[11],[12]]])
    for state in states:
        np.testing.assert_array_equal(state['window'].ravel(),[10,11,12,4,5,6,7])
        np.testing.assert_array_equal(state['compressor'][0].ravel(),[20,24,25,26,27])
        np.testing.assert_array_equal(state['compressor'][1].ravel(),[30,34,35,36,37])


@pytest.mark.parametrize('ids',[np.zeros((2,1)),np.zeros((1,0)),np.zeros((1,9)),np.zeros((1,1,1))])
def test_invalid_geometry_fails_before_any_target_call(ids):
    model=Model()
    with pytest.raises(ValueError,match='one to eight'): forward(model,[ids],[[Cache()]])
    assert model.calls==[]


def test_cache_validation_precedes_all_requests():
    model=Model();bad=[Cache(),Cache()];bad[1].offset=4
    with pytest.raises(ValueError,match='aligned'): forward(model,[np.ones((1,1)),np.ones((1,1))],[[Cache()],bad])
    assert model.calls==[]


def test_tensor_parallel_is_rejected_before_target_execution():
    model=Model();model.pipeline_layers=[SimpleNamespace(_tensor_group=object())]
    with pytest.raises(ValueError,match='local model'):forward(model,[np.ones((1,1))],[[Cache()]])
    assert model.calls==[]


def test_mid_block_failure_clears_partial_stash_and_requires_cache_discard():
    class FailedModel(Model):
        def __call__(self,*args,**kwargs):
            if self.calls:raise RuntimeError('injected target failure')
            return super().__call__(*args,**kwargs)
    model=FailedModel();cache=[Cache()];ids=np.array([[4,5]])
    with pytest.raises(RuntimeError,match='injected'):forward(model,[ids],[cache])
    assert cache[0]._mtp_draft_stash is None
    assert cache[0]._mtp_exact_failed is True
    with pytest.raises(ValueError,match='Discard'):forward(Model(),[ids],[cache])
