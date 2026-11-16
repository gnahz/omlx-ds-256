# Bounded DSpark draft batching

The portable DeepSeek profile enables `DS41_DRAFT_BATCH=1` and declares `effective_settings.dspark_draft_batch=true`. This allows compatible requests to share the three-stage MTP proposal body. Each request retains its own preparation, adaptive draft controller, sampling, cache and completion order. Packed affine vocabulary heads keep their original per-request `(1, width, hidden)` call shape.

The feature does not batch every concurrent request. Eligibility is checked on each draft cycle, and unsupported requests or groups use the existing per-request path. MTP itself must be enabled and available. The portable profile keeps adaptive maximum depth four, batched verification and batch projections enabled; it does not enable the experimental grouped-expert or cross-request attention paths.

## Admission and fallback

| Condition | Requirement |
| --- | --- |
| Sampling | Greedy (`temperature=0`), or the restricted temperature-one path below |
| Temperature-one sampler | Exactly `temperature=1`, `top_p=1`, `top_k=0`, `min_p=0`, `xtc_probability=0`, with the standard MTP sampling hooks |
| Temperature-one processors | No actual processor except the model's static suppression processor marked `_omlx_suppress_processor=True`; no active CopyIndex |
| Request route | Local ordinary DSpark; no external drafter, distributed/cluster MTP, or generic context-copy mode |
| Cache boundary | The cycle must be far enough from the request's aligned cache-commit boundary; boundary cycles use the original path |
| Group | Same model/host object and same current draft width, width 2–4, at least two requests, at most eight total draft rows |
| Device and projections | MLX GPU; affine 8-bit/group-64 vocabulary head without input quantization; supported quantized MTP projections with input quantization and bfloat16 `wo_a` weights |

For width two, a group can contain up to four requests; for width three or four, up to two. Here **width means current proposal width**, not prompt length. Prompt content, prompt length and seed do not need to match: the group key is model/host identity plus draft width. Compatible greedy and temperature-one requests can share a group. Requests still need to reach compatible draft cycles together; four admitted request slots do not guarantee a four-request draft group.

The temperature-one gate checks the **effective sampler and constructed processors**, after request/model/default settings are resolved. An active repetition, presence or frequency penalty, no-repeat-ngram processor, or thinking-budget processor excludes that request from shared temperature-one drafting. Default neutral penalties do not construct those penalty processors. Static model token suppression is the explicit exception. The greedy branch is not subject to this temperature-one processor gate, but still has to pass the other MTP and grouping checks.

A compiled JSON/schema grammar installs a grammar processor and already disables MTP in the existing request route. If JSON mode falls back to a prompt instruction because a grammar cannot be compiled, that instruction alone is not a processor and does not exclude draft batching. The current OpenAI-compatible request path does not wire `logit_bias` into the sampler; supplying that field is not evidence that a bias processor ran. If a non-suppression processor is supplied through another integration, it fails the temperature-one processor gate.

Width-one groups, unmatched widths, unsupported projection formats and other admission failures retain the per-request proposal path. Copy-only or zero-depth preparation remains request-local. This changes proposal construction, not memory admission: the 240 GiB process cap, 238 GiB prefill target, 16 GiB system budget and eight-row draft workspace bound remain enforced.

## Inspect actual use

The authenticated `GET /api/runtime-profile` response exposes:

- `actual.draft_batch.enabled`, `diagnostics_enabled` and `max_rows`;
- `actual.draft_batch.stats.counts`, including `shared_forwards`, `shared_requests`, `shared_stochastic_requests`, `shared_rows` and `singleton_forwards`;
- `actual.draft_batch.stats.groups` for group shapes such as `2x4`, and `fallbacks` for refusal reasons;
- `actual.experiment_controls` for the verification/projection switches and inactive experiment controls.

The authenticated `GET /api/v41-optimizations` response also includes `actual`, but its weight inventory requires an idle engine and returns HTTP 409 while the engine is busy. Use the lighter runtime-profile endpoint for counters during normal operation.

Counters accumulate in the service process and may be absent before their first event. Compare before/after snapshots: `enabled=true` establishes permission to use the path, while an increase in `shared_stochastic_requests` establishes that temperature-one draft sharing occurred. A singleton workload has no shared forwards. Any recorded host construction times are not synchronized GPU timings; no extra GPU synchronization is added for these counters.

## Upgrading an existing portable installation

The launcher uses the checked profile, clears inherited optimization environment variables, and verifies that its actual draft-batching flag agrees with the declared setting. Exporting `DS41_DRAFT_BATCH` in a shell does not override the portable policy.

`git pull` does not rewrite `$OMLX_DS_STATE/profile.json` or change an already running process. `portable.validate_configuration` compares the complete `effective_settings`, `environment` and other preserved policy fields against the template before loading weights. A profile created before this update is rejected on the next start because those fields differ. Follow the existing configure rule: stop this portable service before changing its configuration.

In your own maintenance window, keep the original state directory, model path, metadata proof, host and port, then run:

```sh
# Use the same OMLX_DS_STATE and original installation values.
./deploy/deepseek/service.sh stop
./deploy/deepseek/service.sh configure \
  --model-path "$MODEL_DIR" \
  --metadata-proof "$METADATA_DIR/metadata-proof.json" \
  --host "$SERVICE_HOST" --port "$SERVICE_PORT"
./deploy/deepseek/service.sh start
```

Set `SERVICE_HOST` and `SERVICE_PORT` to the original configured values before running the commands. Omitting them selects the configure defaults, `127.0.0.1:8080`. Configuration preserves an existing nonempty API key. Public lifecycle script paths remain unchanged. This procedure concerns the portable installation owned by that state directory; it does not migrate or operate a separate source-runtime service.

## Prior source-runtime A/B evidence (2026-11-16)

These measurements were completed on the existing source-runtime installation before this repository sync: Apple M5 Ultra, 256 GiB unified memory, macOS 27.0.1, Python 3.13 and the pinned custom MLX 0.32.4. They did not test a fresh install of this portable package.

The test used code and prose workloads with the same prompt within each wave, at concurrency N=1, 2 and 4. Each request had 8,193 input tokens, an 8,192-token warm prefix hit and exactly 1,024 output tokens ending with `length`. Sampling was temperature one, top-p one and top-k/min-p zero. A disabled draft sharing and B enabled it in the same service process; the model, adaptive MTP settings and memory guards stayed fixed. Each of the six cells ran two ABBA rounds: **48 timed waves and 112 timed requests**. One cold seed request per workload and 128-token warmups for both arms of each cell were excluded from timing. API usage and native completion token/cache counts agreed for all timed requests.

Throughput below is aggregate output tokens divided by the sum of each wave's common native generation interval, from the earliest request's `generation_started_at` to the last completion callback entry. The measurement added one CPU completion timestamp per request, without a per-token hook or GPU synchronization. It is not the average of individual request rates. Paired A/B requests used matching `42 + row_index + round_index` seeds.

| Workload | N | A: draft sharing off (tokens/s) | B: draft sharing on (tokens/s) | Observed difference |
| --- | ---: | ---: | ---: | ---: |
| Code | 1 | 59.200 | 59.402 | +0.34% |
| Code | 2 | 66.850 | 67.850 | +1.49% |
| Code | 4 | 71.806 | 74.307 | +3.48% |
| Prose | 1 | 49.086 | 49.233 | +0.30% |
| Prose | 2 | 57.913 | 59.292 | +2.38% |
| Prose | 4 | 62.734 | 64.705 | +3.14% |

All A waves and all N=1 waves recorded zero shared stochastic requests. Every concurrent B wave recorded actual temperature-one sharing. N=1 therefore serves as a control; its small differences are not evidence of draft-batching benefit.

The eight N=1 A/B output pairs matched, while all 48 concurrent A/B output pairs differed despite paired seed values. Logged MTP statistics were matched to native completion UIDs for all 112 timed requests, with no missing or duplicate relevant UIDs. They also show different token paths and acceptance rates; for example, prose N=2 checked-prefix acceptance changed from 59.37% to 61.07%. This log ratio is accepted tokens divided by checked prefix positions, where checking stops at the first rejection; it is not acceptance over all submitted proposals.

The table is descriptive evidence from two rounds per cell, not a same-output comparison, a statistical-significance result, a distributional-equivalence proof or a general performance guarantee. Sampling and MTP acceptance can confound small throughput differences. Prose N=2, for example, measured +4.29% and +0.48% in its two rounds. Different-prompt, longer-context and arbitrary sampling settings are outside this test. Earlier temperature-zero measurements are separate evidence and were not used as the temperature-one baseline.

The full lifecycle observer reported normal pressure, **zero cumulative new swap-in plus swap-out**, and **zero peak growth of compressor physical occupancy above its starting baseline**, under the existing 100 MiB paging and 2 GiB compressor-growth allowances. This does not mean the system had no pre-existing compressed memory or swap, or no logical compression activity. The observer took 4,035 samples at a nominal 0.5-second interval; peaks between samples remain unverified. The source service was restored after that run. No service or API operation was performed to publish this documentation.

Only aggregate results are included here. Private prompts, generated responses, absolute installation paths, per-request identifiers and large runtime logs are not part of the repository. See [validation.md](validation.md) for the separate CPU checks performed for this source sync and the remaining release limitations.
