"""Bounded ordinary-DSpark draft batching with explicit startup opt-in.

No MLX import or payload allocation at module import. The scheduler opts in only
for greedy or default-temperature local requests away from cache boundaries.
Every request retains its own prepare, CopyIndex, controller depth, sample and
cache history. Only the three-stage proposal body is shared. Packed affine
vocabulary heads keep exactly the singleton (1, width, hidden) call shape.
"""
from collections import Counter
from copy import deepcopy
import time

_STATS = Counter()
_GROUPS = Counter()
_FALLBACKS = Counter()


def stats(reset=False):
    result = {"counts": dict(_STATS), "groups": dict(_GROUPS),
              "fallbacks": dict(_FALLBACKS), "maximum_group_rows": 8,
              "supported_temperatures": [0.0, 1.0],
              "stochastic_scope": "temperature=1, top_p=1, top_k=min_p=xtc=0; standard sampler; no stateful processors",
              "heads_keep_request_shape": True,
              "timing": "Host construction time only; no additional GPU synchronization"}
    if reset:
        _STATS.clear(); _GROUPS.clear(); _FALLBACKS.clear()
    return deepcopy(result)


def stochastic_reason(row, state, bg):
    """Admit only the existing standard, unfiltered temperature-one sampler."""
    sampler = bg._resolve_sampler(row)
    if not (getattr(sampler, "temp", None) == 1.0
            and getattr(sampler, "top_p", None) == 1.0
            and getattr(sampler, "top_k", None) == 0
            and getattr(sampler, "min_p", None) == 0.0
            and getattr(sampler, "xtc_probability", None) == 0.0
            and callable(getattr(sampler, "_mtp_sampling_logits", None))
            and callable(getattr(sampler, "sample_with_logprobs", None))):
        return "unsupported_stochastic_sampler"
    if any(getattr(proc, "_omlx_suppress_processor", False) is not True
           for proc in (bg._proc_list(row) or [])):
        return "stochastic_processors"
    if getattr(state, "copy_index", None) is not None:
        return "stochastic_copy_index"
    return None


def can_defer(row, state, bg):
    """Conservative scheduler gate before any request-local state is changed."""
    reason = None
    if not bg._is_greedy(row):
        reason = stochastic_reason(row, state, bg)
    if reason:
        _FALLBACKS[reason] += 1
        return False
    if bg._drafter_for(row.model) is not None:
        reason = "external_drafter"
    elif bg._cluster_mtp(row.model) is not None:
        reason = "distributed_request"
    elif bg.DSPARK_GENERIC_CONTEXT_COPY:
        reason = "generic_context_copy"
    else:
        alignment = int(getattr(row.model, "_omlx_mtp_commit_align", 0) or 0)
        emitted = len(row.tokens[0])
        remaining = ((emitted // alignment) + 1) * alignment - emitted if alignment else 0
        if alignment and remaining <= 1 + int(state.drafts.shape[0]):
            reason = "commit_boundary"
    if reason:
        _FALLBACKS[reason] += 1
        return False
    return True


def group_plans(plans, max_rows=8):
    """Stable, bounded cohorts by host identity and width (no MLX work)."""
    if max_rows != 8:
        raise ValueError("The validated DSpark workspace is bounded to eight rows")
    buckets = {}
    groups = []
    for index, plan in enumerate(plans):
        if plan is None:
            continue
        host, width, *_ = plan
        if not isinstance(width, int) or not 2 <= width <= 4:
            groups.append([index])
            continue
        key = (id(host), width)
        bucket = buckets.setdefault(key, [])
        bucket.append(index)
        if len(bucket) * width + width > max_rows:
            groups.append(bucket[:])
            bucket.clear()
    groups.extend(bucket for bucket in buckets.values() if bucket)
    # Forward construction can be grouped; sampling below retains original order.
    return sorted(groups, key=lambda group: group[0])


def run_jobs(jobs, depths, *, bg, dspark):
    """Prepare once, group compatible proposal bodies, then finish in UID order.

    bg/dspark are passed explicitly so control-flow and ownership invariants are
    testable without importing MLX or allocating any model weights.
    """
    if len(jobs) != len(depths):
        raise ValueError("Each deferred draft must retain its pre-observe controller depth")
    _STATS["job_batches"] += 1
    _STATS["jobs"] += len(jobs)
    plans = []
    for (row, state, hidden, committed, previous), depth in zip(jobs, depths):
        began = time.perf_counter()
        if not state.head_clone:
            bg._mtp_head_trim_to(state.mtp_cache, state.hist_offset)
        controller = state.controller
        observed = controller.cur if controller is not None else None
        if controller is not None and depth is not None:
            controller.cur = depth
        try:
            plans.append(bg._dspark_prepare(row, state, hidden, committed, previous))
        finally:
            if controller is not None and depth is not None:
                controller.cur = observed
        state.stats.mtp_head_ms += (time.perf_counter() - began) * 1000
    _STATS["copy_or_zero_depth_jobs"] += sum(plan is None for plan in plans)
    outputs = {}
    shares = {}
    for group in group_plans(plans):
        host, width, *_ = plans[group[0]]
        began = time.perf_counter()
        widths = [plans[index][1] for index in group]
        reason = dspark.candidate_batch_reason(host, widths)
        if reason is None:
            for index in group:
                host.dspark_append_context(jobs[index][2], jobs[index][1].mtp_cache)
            blocks = dspark.proposal_forward_batch(
                host, [plans[index][2] for index in group],
                [jobs[index][1].mtp_cache for index in group], width,
                [bool(plans[index][3]) for index in group],
                preserve_head_shapes=True,
            )
            if len(blocks) != len(group):
                raise RuntimeError("Batched DSpark returned the wrong number of request outputs")
            _STATS["shared_forwards"] += 1
            _STATS["shared_requests"] += len(group)
            _STATS["shared_stochastic_requests"] += sum(
                not bg._is_greedy(jobs[index][0]) for index in group
            )
            _STATS["shared_rows"] += len(group) * width
            _GROUPS[f"{len(group)}x{width}"] += 1
        else:
            _FALLBACKS[reason] += 1
            blocks = [
                plans[index][0].dspark_forward(
                    jobs[index][2], plans[index][2], jobs[index][1].mtp_cache,
                    draft_length=plans[index][1],
                    **bg._dspark_head_kwargs(plans[index][0], bool(plans[index][3])),
                )[0]
                for index in group
            ]
            _STATS["singleton_forwards"] += len(group)
        share = (time.perf_counter() - began) * 1000 / len(group)
        for index, block in zip(group, blocks):
            outputs[index] = block
            shares[index] = share
    for index, (row, state, _, committed, previous) in enumerate(jobs):
        if plans[index] is None:
            continue
        began = time.perf_counter()
        state.hist_offset += int(committed.shape[0])
        bg._dspark_finish(row, state, plans[index], outputs[index], previous)
        state.stats.mtp_head_ms += (time.perf_counter() - began) * 1000 + shares[index]
    return bool(outputs)
