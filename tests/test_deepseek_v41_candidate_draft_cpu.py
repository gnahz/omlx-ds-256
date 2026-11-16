"""CPU-only orchestration checks: imports neither MLX nor any model payload."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

SOURCE = (Path(__file__).resolve().parents[1]
          / "omlx" / "patches" / "deepseek_v41" / "candidate_draft.py")
spec = importlib.util.spec_from_file_location("candidate_draft_cpu", SOURCE)
candidate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(candidate)


@pytest.fixture(autouse=True)
def isolated_stats():
    candidate.stats(reset=True)
    yield
    candidate.stats(reset=True)


def test_groups_never_mix_hosts_widths_or_exceed_workspace():
    one, two = object(), object()
    plans = [(one, 4, None, False), (one, 2, None, False), None,
             (one, 4, None, False), (two, 4, None, False),
             (one, 4, None, False), (one, 2, None, False),
             (one, 1, None, False), (one, 5, None, False)]
    groups = candidate.group_plans(plans)
    assert groups == [[0, 3], [1, 6], [4], [5], [7], [8]]
    assert sorted(i for group in groups for i in group) == [0, 1, 3, 4, 5, 6, 7, 8]
    for group in groups:
        assert len({id(plans[i][0]) for i in group}) == 1
        assert len({plans[i][1] for i in group}) == 1
        if len(group) > 1:
            assert sum(plans[i][1] for i in group) <= 8
    with pytest.raises(ValueError, match="eight"):
        candidate.group_plans(plans, max_rows=16)


@pytest.mark.parametrize("reason", ["unsupported_stochastic_sampler", "external_drafter",
                                      "distributed_request", "generic_context_copy", "commit_boundary"])
def test_unsafe_rows_are_not_deferred(reason):
    row = NS(model=NS(_omlx_mtp_commit_align=8192), tokens=[[None] * 8188])
    state = NS(drafts=NS(shape=(4,)))
    bg = NS(_resolve_sampler=lambda row: NS(temp=0.7), _is_greedy=lambda row: reason != "unsupported_stochastic_sampler",
            _drafter_for=lambda model: object() if reason == "external_drafter" else None,
            _cluster_mtp=lambda model: object() if reason == "distributed_request" else None,
            DSPARK_GENERIC_CONTEXT_COPY=reason == "generic_context_copy")
    assert not candidate.can_defer(row, state, bg)
    if reason == "commit_boundary":
        row.tokens = [[None] * 8000]
        assert candidate.can_defer(row, state, bg)


def test_jobs_preserve_controller_depth_prepare_once_copy_and_finish_order():
    candidate.stats(reset=True)
    events = []

    class Host:
        def dspark_append_context(self, hidden, cache):
            events.append(("append", hidden))
            cache.append(hidden)

        def dspark_forward(self, hidden, anchor, cache, *, draft_length, cost_policy=False):
            self.dspark_append_context(hidden, cache)
            events.append(("single", hidden, draft_length))
            return ("logits", hidden), None

    host = Host()
    jobs, depths = [], [4, 2, 4, 2, 4]
    for index in range(5):
        state = NS(controller=NS(cur=1), head_clone=False, mtp_cache=[], hist_offset=10,
                   stats=NS(mtp_head_ms=0), is_copy=index == 4)
        jobs.append((index, state, index, NS(shape=(index+1,)), f"prev{index}"))

    def prepare(row, state, hidden, committed, previous):
        events.append(("prepare", row, state.controller.cur, previous))
        if state.is_copy:
            host.dspark_append_context(hidden, state.mtp_cache)
            state.hist_offset += committed.shape[0]
            return None
        return host, state.controller.cur, row, False

    def finish(row, state, plan, logits, previous):
        assert state.controller.cur == 1
        assert logits == ("logits", row)
        assert previous == f"prev{row}"
        events.append(("finish", row))

    bg = NS(_is_greedy=lambda row: True, _mtp_head_trim_to=lambda cache, offset: events.append(("trim", offset)),
            _dspark_prepare=prepare, _dspark_finish=finish,
            _dspark_head_kwargs=lambda host, policy: {"cost_policy": policy})

    def forward_batch(model, anchors, caches, width, policies, *, preserve_head_shapes):
        assert preserve_head_shapes is True
        assert len(anchors) * width <= 8
        events.append(("batch", tuple(anchors), width))
        return [("logits", anchor) for anchor in anchors]

    dsp = NS(candidate_batch_reason=lambda model, widths: None if len(widths) > 1 else "singleton",
             proposal_forward_batch=forward_batch)
    assert candidate.run_jobs(jobs, depths, bg=bg, dspark=dsp)
    assert [e for e in events if e[0] == "batch"] == [("batch", (0, 2), 4), ("batch", (1, 3), 2)]
    assert [e for e in events if e[0] == "finish"] == [("finish", i) for i in range(4)]
    assert len([e for e in events if e[0] == "prepare"]) == 5
    for index, (_, state, *_) in enumerate(jobs):
        assert state.hist_offset == 11 + index
        assert state.mtp_cache == [index]
    counts = candidate.stats()["counts"]
    assert counts["shared_forwards"] == 2
    assert counts["shared_rows"] == 12
    assert counts["copy_or_zero_depth_jobs"] == 1


def test_failed_prepare_restores_controller_and_rejects_missing_depth():
    state = NS(controller=NS(cur=2), head_clone=True, mtp_cache=[], hist_offset=0,
               stats=NS(mtp_head_ms=0))
    job = (None, state, None, NS(shape=(1,)), None)
    def fail(*args):
        assert state.controller.cur == 4
        raise ValueError("expected")
    with pytest.raises(ValueError, match="expected"):
        candidate.run_jobs([job], [4], bg=NS(_dspark_prepare=fail), dspark=None)
    assert state.controller.cur == 2
    with pytest.raises(ValueError, match="controller depth"):
        candidate.run_jobs([job], [], bg=None, dspark=None)


def test_singleton_and_unsupported_groups_preserve_original_forward():
    calls = []
    class Host:
        def dspark_forward(self, hidden, anchor, cache, *, draft_length):
            calls.append((hidden, draft_length))
            return hidden, None
    host = Host()
    jobs = [(i, NS(head_clone=True, controller=None, mtp_cache=[], hist_offset=0,
                   stats=NS(mtp_head_ms=0)), i, NS(shape=(1,)), None) for i in range(2)]
    bg = NS(_dspark_prepare=lambda row, *args: (host, 4, row, False),
            _dspark_head_kwargs=lambda *args: {}, _dspark_finish=lambda *args: None)
    candidate.run_jobs(jobs, [None, None], bg=bg,
        dspark=NS(candidate_batch_reason=lambda *args: "head_format"))
    assert calls == [(0, 4), (1, 4)]


def eligible(temp=1., **kw):
    params=dict(temp=temp, top_p=1., top_k=0, min_p=0., xtc_probability=0.,
                _mtp_sampling_logits=lambda x:x, sample_with_logprobs=lambda x:x)
    params.update(kw)
    sampler=NS(**params)
    row=NS(model=NS(_omlx_mtp_commit_align=8192),tokens=[[0]*8000])
    state=NS(drafts=NS(shape=(4,)),copy_index=None)
    bg=NS(_is_greedy=lambda row:temp==0, _resolve_sampler=lambda row:sampler,
          _proc_list=lambda row:None,_drafter_for=lambda model:None,
          _cluster_mtp=lambda model:None,DSPARK_GENERIC_CONTEXT_COPY=False)
    return row,state,bg


def test_default_temperature_one_and_greedy():
    for temp in (0.,1.):
        assert candidate.can_defer(*eligible(temp))


@pytest.mark.parametrize("kw",[dict(temp=.7),dict(temp=2.),dict(top_p=.95),dict(top_k=20),
    dict(min_p=.05),dict(xtc_probability=.1),dict(sample_with_logprobs=None),dict(_mtp_sampling_logits=None)])
def test_other_stochastic_samplers_fall_back(kw):
    assert not candidate.can_defer(*eligible(**kw))


def test_processor_and_copy_ownership_and_boundary():
    row,state,bg=eligible(); bg._proc_list=lambda row:[NS(_omlx_suppress_processor=True)]
    assert candidate.can_defer(row,state,bg)
    bg._proc_list=lambda row:[object()]
    assert not candidate.can_defer(row,state,bg)
    bg._proc_list=lambda row:None; state.copy_index=object()
    assert not candidate.can_defer(row,state,bg)
    state.copy_index=None; row.tokens=[[0]*8188]
    assert not candidate.can_defer(row,state,bg)


def test_stochastic_keeps_lifecycle_fallbacks():
    for kind in ("external","distributed","generic"):
        row,state,bg=eligible()
        if kind=="external":bg._drafter_for=lambda model:object()
        if kind=="distributed":bg._cluster_mtp=lambda model:object()
        if kind=="generic":bg.DSPARK_GENERIC_CONTEXT_COPY=True
        assert not candidate.can_defer(row,state,bg)


@pytest.mark.parametrize("width", [2, 3, 4])
def test_over_capacity_groups_cover_every_request_without_exceeding_eight_rows(width):
    host = object()
    plans = [(host, width, index, False) for index in range(11)]
    groups = candidate.group_plans(plans)
    assert [index for group in groups for index in group] == list(range(11))
    assert all(0 < len(group) * width <= 8 for group in groups)
    assert all((len(group) + 1) * width > 8 for group in groups[:-1])


def test_mixed_greedy_and_temperature_one_share_body_but_finish_in_request_order():
    events = []

    class Host:
        _omlx_mtp_commit_align = 8192

        def dspark_append_context(self, hidden, cache):
            cache.append(hidden)

    host = Host()
    jobs = []
    for index, temperature in enumerate((1.0, 0.0, 1.0, 0.0)):
        row, state, original_bg = eligible(temperature)
        row.model = host
        row.index = index
        row.temperature = temperature
        row.sampler = original_bg._resolve_sampler(row)
        state.controller = NS(cur=1)
        state.head_clone = True
        state.mtp_cache = []
        state.hist_offset = 10
        state.stats = NS(mtp_head_ms=0)
        jobs.append((row, state, f"hidden-{index}", NS(shape=(1,)), f"previous-{index}"))

    def prepare(row, state, hidden, committed, previous):
        assert state.controller.cur == 2
        events.append(("prepare", row.index))
        return host, 2, row.index, False

    def finish(row, state, plan, logits, previous):
        assert state.controller.cur == 1
        assert logits == f"logits-{row.index}"
        assert previous == f"previous-{row.index}"
        assert state.mtp_cache == [f"hidden-{row.index}"]
        events.append(("finish", row.index, row.temperature))

    bg = NS(
        _is_greedy=lambda row: row.temperature == 0.0,
        _resolve_sampler=lambda row: row.sampler,
        _proc_list=lambda row: None,
        _drafter_for=lambda model: None,
        _cluster_mtp=lambda model: None,
        DSPARK_GENERIC_CONTEXT_COPY=False,
        _dspark_prepare=prepare,
        _dspark_finish=finish,
    )
    for row, state, *_ in jobs:
        assert candidate.can_defer(row, state, bg)

    def forward_batch(model, anchors, caches, width, policies, *, preserve_head_shapes):
        assert model is host
        assert anchors == [0, 1, 2, 3]
        assert width == 2
        assert preserve_head_shapes is True
        assert policies == [False] * 4
        events.append(("batch", tuple(anchors)))
        return [f"logits-{anchor}" for anchor in anchors]

    dsp = NS(candidate_batch_reason=lambda model, widths: None,
             proposal_forward_batch=forward_batch)
    assert candidate.run_jobs(jobs, [2] * 4, bg=bg, dspark=dsp)
    assert events == [
        *(('prepare', index) for index in range(4)),
        ('batch', (0, 1, 2, 3)),
        ('finish', 0, 1.0), ('finish', 1, 0.0),
        ('finish', 2, 1.0), ('finish', 3, 0.0),
    ]
    assert all(state.hist_offset == 11 for _, state, *_ in jobs)
    snapshot = candidate.stats()
    assert snapshot['groups'] == {'4x2': 1}
    assert snapshot['counts']['shared_requests'] == 4
    assert snapshot['counts']['shared_stochastic_requests'] == 2
    assert snapshot['counts']['shared_rows'] == 8


def test_stats_snapshots_are_detached_and_reset_returns_prior_counts():
    assert not candidate.can_defer(*eligible(temp=0.7))
    snapshot = candidate.stats()
    assert snapshot['fallbacks'] == {'unsupported_stochastic_sampler': 1}
    snapshot['fallbacks']['unsupported_stochastic_sampler'] = 999
    snapshot['counts']['jobs'] = 999
    snapshot['groups']['4x2'] = 999
    snapshot['supported_temperatures'].append(2.0)
    before_reset = candidate.stats(reset=True)
    assert before_reset['fallbacks'] == {'unsupported_stochastic_sampler': 1}
    assert before_reset['counts'] == {}
    assert before_reset['groups'] == {}
    assert before_reset['supported_temperatures'] == [0.0, 1.0]
    cleared = candidate.stats()
    assert cleared['fallbacks'] == cleared['counts'] == cleared['groups'] == {}


@pytest.mark.parametrize('reason', ['disabled', 'device', 'head_format', 'stage_format', 'workspace'])
def test_refused_batch_preserves_every_singleton_and_reports_fallback(reason):
    forwards, finishes = [], []

    class Host:
        def dspark_forward(self, hidden, anchor, cache, *, draft_length):
            cache.append(hidden)
            forwards.append((hidden, anchor, draft_length))
            return f'logits-{anchor}', None

    host = Host()
    jobs = [(index, NS(head_clone=True, controller=None, mtp_cache=[], hist_offset=0,
                       stats=NS(mtp_head_ms=0)), index, NS(shape=(1,)), None)
            for index in range(2)]

    def finish(row, state, plan, logits, previous):
        assert state.mtp_cache == [row]
        assert state.hist_offset == 1
        finishes.append((row, logits))

    bg = NS(_dspark_prepare=lambda row, *args: (host, 4, row, False),
            _dspark_head_kwargs=lambda *args: {}, _dspark_finish=finish)
    assert candidate.run_jobs(jobs, [None, None], bg=bg,
        dspark=NS(candidate_batch_reason=lambda *args: reason))
    assert forwards == [(0, 0, 4), (1, 1, 4)]
    assert finishes == [(0, 'logits-0'), (1, 'logits-1')]
    snapshot = candidate.stats()
    assert snapshot['fallbacks'] == {reason: 1}
    assert snapshot['counts']['singleton_forwards'] == 2
    assert snapshot['counts'].get('shared_forwards', 0) == 0
