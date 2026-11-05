# Validation and release scope

## Source and packaging checks (2026-11-05)

The fork starts at official oMLX v0.7.3, commit `9de79bc39a01c1f3fec1e849903aaf0dd5478d7a`. The import covers all 200 modified or added source files in the local 0.7.3 customization set, including scheduler/backend work, native kernels, frontend templates, locale updates, and regression tests. The separate deployment helpers and the nine-file MLX patch are also included.

[`source-manifest.json`](source-manifest.json) records the original imported file hashes and the final portable file hashes. The portability changes replace machine-specific test paths, provide a checked-in architecture-only fixture, make external tokenizer tests explicitly optional, use loopback defaults for optional remote-encoder code, and update an obsolete cache test fixture to the current metadata format. Runtime numerical behavior and memory ceilings are preserved. The dependency declaration comment now explains why the stock same-version MLX wheel is insufficient.

Checks performed on the independent checkout:

- 146 CPU source regression tests passed, with two expected skips for optional tokenizer/model fixtures. The selected tests cover cancellation, aligned I/O, exact-cache validation, native dispatch/workspace planning, bounded prefill peak estimates, NAX capability checks, and reservation resizing. They do not import MLX or load weights.
- Model-download and exact-metadata preparation: 26 CPU tests passed using small synthetic files, including checksums, identity drift, hardlinks, unchanged source bytes, path containment, corruption, and header/config mismatches.
- 17 portable lifecycle and configuration tests passed using mocks. They exercise ownership, refusal of unrelated PIDs/listeners, detached launch configuration, readiness, shutdown/restart ordering, external state, and recorder job definitions. They do not launch an inference service or change launchd state.
- Two existing Node SSD-cache UI action tests passed. All imported Python, JSON, and Jinja templates parse; the dashboard JavaScript syntax check and Git whitespace checks passed. This packaging pass did not rerun the browser-based dashboard suite.
- The MLX patch applies cleanly to a fresh archive of its pinned base. All nine resulting files match the original custom MLX source byte for byte. No native binaries are shipped.
- The pinned Hugging Face download preview succeeded without downloading weights. All 250 tensor LFS hashes agree with the published SHA256SUMS. Sizes and header hashes of the local 1017 checkpoint/overlay match the public metadata; only 252,741 header bytes were read for that comparison.

The portable bootstrap/native build and a fresh model download/load have **not** been executed in this packaging pass. Build-script validation does not establish native ABI compatibility on another machine. No production API endpoint, service lifecycle, or model inference was used for these checks.

## Reproduce CPU checks

Create a development environment separate from the deployment environment:

```sh
uv venv --python 3.13 .venv-dev
uv pip install --python .venv-dev/bin/python pytest numpy jinja2 psutil
```

Use `--noconftest` because the full upstream test configuration imports MLX. These selected files are intentionally CPU-only:

```sh
.venv-dev/bin/python -m pytest --noconftest -q \
  tests/test_astra_cancel.py \
  tests/test_deepseek_v41_aligned_io_cpu.py \
  tests/test_deepseek_v41_exact_cache_io_cpu.py \
  tests/test_deepseek_v41_exact_verify_cpu.py \
  tests/test_deepseek_v41_native_gate_up_cpu.py \
  tests/test_nax_hardware_cpu.py \
  tests/test_scheduler_prefill_peak_target_cpu.py \
  tests/test_prefill_budget_resize.py \
  tests/test_whole_prefill_peak_cpu.py \
  tests/test_deepseek_v41_native_workspace_cpu.py \
  tests/test_og_images.py \
  tests/test_deepseek_v41_fast_encode.py

.venv-dev/bin/python -m pytest --noconftest -q deploy/deepseek/tests
.venv-dev/bin/python -m unittest discover -s tests/deployment -v
node --test apps/omlx-web/tests/ssd_cache_clear_ui.test.cjs
```

Leave `DS41_TEST_MODEL_DIR` unset for this CPU packaging run. The optional tokenizer regression also needs its external `fe_conv` generator; it skips when that fixture is unavailable. The full test suite includes real MLX/GPU and optional external-model tests and is not an appropriate default for source-only maintenance beside an active deployment.

## Prior source-runtime evidence

The source deployment, before portable packaging, was tested on an Apple M5 Ultra, 256 GiB unified memory, macOS 27.0.1. Its validation covered the migrated frontend, cache behavior, sampling/MTP, image requests, four concurrent shorter requests, cold 200K context, cache persistence, and service stop/memory release. Those results describe the original installation; they are not a fresh-install certification of this repository.

One separate cold multimodal run completed 372,432 prompt tokens, 11 images, and 4,096 output tokens. It reached about 229.88 GiB process footprint, about 230.23 MiB peak compressor growth, and zero new swap I/O. First output arrived in about 255.7 seconds. This is one isolated long-context sample, not a concurrency guarantee or a statistical benchmark.

Known limits remain explicit:

- Several simultaneous long contexts can exhaust physical headroom. A previous three-decode/one-waiting workload exceeded the 2 GiB compressor-growth acceptance despite remaining below the process cap. This fork does not claim that case is fixed or widen its acceptance limits.
- A long image-bearing prefill can hold a scheduler step for an extended period while the HTTP loop remains responsive. The prefill defer setting is not a maximum decode-latency guarantee.
- The default ragged verification comparison has four known numerical failures also present in the 0.7.1 baseline (10 of 14 comparisons passed). An optional stricter verification path passed its targeted checks but was substantially slower and is not selected by the deployment profile. The migration is not claimed to be bitwise identical for every inference path.
- The independent LSQ3/G128 A8 prefill experiment showed no benefit on this hardware; its sampled kernels were slower. It is not enabled or promoted as a performance improvement.
- A configured 1M-token context limit is a ceiling, not evidence that every request of that size is admissible on 256 GiB.

For new heavyweight validation, use an explicitly scheduled window, preserve all memory guards, stop competing runtimes only with authorization, and record the whole lifecycle using the observer described in the README.
