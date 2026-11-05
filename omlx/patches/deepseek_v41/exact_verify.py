# SPDX-License-Identifier: MIT
"""Optional singleton-arithmetic verification for diagnosing short-block drift.

The default batched verifier remains unchanged. This path uses the existing
one-token target calls, while joining their bounded undo rows so the normal
partial-rollback implementation can commit any verified prefix without replay.
On failure, all supplied request caches must be discarded; this diagnostic does
not claim transactional rollback of mutable growth buffers.
"""

import mlx.core as mx


def forward(model, inputs, caches):
    if not inputs or len(inputs) != len(caches):
        raise ValueError("One cache is required per verification request")
    if any(ids.ndim != 2 or ids.shape[0] != 1 or not 1 <= ids.shape[1] <= 8
           for ids in inputs):
        raise ValueError("Exact verification supports one to eight unpadded rows per request")
    if getattr(model, "pipeline_size", 1) > 1 or any(
            getattr(layer, "_tensor_group", None) is not None
            for layer in getattr(model, "pipeline_layers", ())):
        raise ValueError("Exact diagnostic verification requires a local model")
    for cache in caches:
        if cache and getattr(cache[0], "_mtp_exact_failed", False):
            raise ValueError("Discard caches from a failed exact verification")
        if not cache or any(item.size() != cache[0].size() for item in cache):
            raise ValueError("Exact verification requires aligned non-empty layer caches")
    try:
        results = []
        for ids, cache in zip(inputs, caches):
            before = cache[0].size()
            snapshots = [(list(item.cache), item.left_padding, item.lengths) for item in cache]
            windows = [[] for _ in cache]
            compressors = [[] for _ in cache]
            logits, hidden = [], []
            for index in range(ids.shape[1]):
                scores, taps = model(ids[:, index:index + 1], cache=cache,
                                     return_hidden=True, target_verify=True)
                logits.append(scores)
                hidden.append(taps)
                step_ids, _, step_before, states = cache[0]._mtp_draft_stash
                if step_ids.shape[1] != 1 or step_before != before + index:
                    raise ValueError("Singleton verification did not return its undo state")
                for layer, state in enumerate(states):
                    # First step includes the original window / incomplete group.
                    # Later steps contribute only their newly projected token.
                    windows[layer].append(state["window"] if index == 0 else state["window"][:, -1:])
                    if "compressor" in state:
                        compressors[layer].append(tuple(value if index == 0 else value[:, -1:]
                                                        for value in state["compressor"]))
            states = []
            for window, compressor in zip(windows, compressors):
                state = {"window": mx.concatenate(window, axis=1)}
                if compressor:
                    state["compressor"] = tuple(mx.concatenate([pair[i] for pair in compressor], axis=1)
                                                  for i in range(2))
                states.append(state)
            cache[0]._mtp_draft_stash = (ids, snapshots, before, states)
            results.append((mx.concatenate(logits, axis=1), mx.concatenate(hidden, axis=1), None))
        return results
    except BaseException:
        # Cache growth can be mutable. Clear misleading partial stashes and
        # reject diagnostic reuse instead of pretending a shallow restore is safe.
        for cache in caches:
            cache[0]._mtp_draft_stash = None
            cache[0]._mtp_exact_failed = True
        raise
