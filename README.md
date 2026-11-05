# omlx-ds-256

DeepSeek V4.1 Flash **1017 LSQ3 / group size 128** on a 256 GiB Apple Silicon Mac, based on **oMLX 0.7.3**.

This fork combines the official [oMLX v0.7.3 release](https://github.com/jundot/omlx/tree/v0.7.3) (`9de79bc39a01c1f3fec1e849903aaf0dd5478d7a`) with the local DeepSeek runtime, native kernels, exact prefix cache, memory admission policy, and dashboard customizations. It is maintained as a standalone source repository. Installation, model downloads, runtime state, and service startup are explicit separate steps.

## What is included

- DeepSeek V4.1 1017 text and vision support, LSQ3/G128 routed experts, MXFP8 backbone projections, quantized embeddings/head, Engram SSD I/O, and MTP drafting.
- Lossless dictionary compression of expert scale/bias metadata, saving about **7.90 GiB of resident memory** without changing quantization codes. Published metadata is prepared locally; weights are not requantized.
- Custom packed sparse attention, bounded native prefill workspace, cache growth, projection and decode paths, plus the pinned custom MLX Q3/G128 gather patch.
- Exact CED prefix caching with image identity, Fibonacci checkpoints, RAM and SSD L2 accounting, and cache controls.
- Whole-prefill memory reservations, queue admission, physical-memory/load guards, and process-footprint limits.
- The custom Active Models dashboard: live request phases, recent output rates, queue positions, memory reservations, pending prefill requirements, and exact-cache statistics. The migrated locale changes are included.
- A portable deployment profile, detached manual lifecycle controls, and memory observation tools under [`deploy/deepseek`](deploy/deepseek).

**A8 prefill is disabled.** The independent LSQ3/G128 A8 experiment did not improve performance on the tested machine and is not selected by this deployment. The custom MLX patch is a Q3/G128 gather implementation, not that A8 prototype.

## Hardware and software

The source runtime was validated on an **Apple M5 Ultra with 256 GiB unified memory**, macOS **27.0.1**, Python **3.13**, custom MLX **0.32.4**, and oMLX **0.7.3**. Other configurations need their own validation. This profile is not sized for smaller-memory Macs.

You need:

- Full Xcode with the Metal compiler/SDK, the command-line developer tools selected, and the Xcode license accepted. The preserved native build targets macOS 27.0; the NAX path additionally needs a compatible GPU and Metal 4 SDK.
- Git, [uv](https://docs.astral.sh/uv/getting-started/installation/), and a fast local SSD.
- About **420.53 GiB** for the complete pinned model download, plus build space and up to **60 GiB** for the default SSD prefix-cache budget. Allow extra space for Hugging Face transfer caches and temporary files.
- Enough free physical memory for roughly **216 GiB** of resident model tensors plus runtime workspaces and the rest of the system. Stop competing model servers yourself before starting this deployment. The scripts never stop unrelated services.

## 1. Clone and build the isolated environment

```sh
git clone https://github.com/gnahz/omlx-ds-256.git
cd omlx-ds-256
export OMLX_DS_STATE="$HOME/.local/share/omlx-ds-256"
uv python install 3.13
./deploy/deepseek/bootstrap.sh
```

The bootstrap builds the pinned custom MLX source and the oMLX native extensions. Dependency commits, ABI pins, and the MLX patch are recorded in [`dependency-lock.json`](deploy/deepseek/dependency-lock.json). This is a build-critical dependency lock, not a lock of every transitive package. The default build parallelism is two jobs.

Keep the checkout at its installed location. Runtime state, the Python environment, build dependencies, logs, API key, settings, and caches live under `OMLX_DS_STATE`; downloaded models live in a separate directory. The bootstrap does **not** start a server or download model weights.

Use this bootstrap for the DeepSeek profile. A generic upstream `pip install`, Homebrew upgrade, or official app install does not reproduce the custom MLX build. Replacing it with a stock wheel having the same version number can remove required kernels.

## 2. Download the correct model

Use the **1017** checkpoint:

- Model: [gnahzhang/DeepSeek-V4.1-Flash-1017-LSQ3-G128-MLX](https://huggingface.co/gnahzhang/DeepSeek-V4.1-Flash-1017-LSQ3-G128-MLX)
- Pinned revision: `c98bd97baca052f484c1d6ea6b6e8a8eced404ef`
- File identities: [`model-manifest.json`](deploy/deepseek/model-manifest.json)

The older model whose name omits `1017` is a different release. The model repository also contains an older runtime bundle; use **this repository's** deployment scripts for oMLX 0.7.3.

```sh
export OMLX_DS_STATE="$HOME/.local/share/omlx-ds-256"
export MODEL_DIR="$HOME/Models/DeepSeek-V4.1-Flash-1017-LSQ3-G128-MLX"
export METADATA_DIR="${MODEL_DIR}-exact-metadata"
export PATH="$OMLX_DS_STATE/venv/bin:$PATH"

# Preview the pinned download without fetching weights.
python deploy/deepseek/download_model.py --model-dir "$MODEL_DIR" --dry-run

# Download the full snapshot and verify its published checksums.
python deploy/deepseek/download_model.py --model-dir "$MODEL_DIR"
```

The helper calls the current `hf download` CLI with the pinned revision and two download workers. Keep the complete snapshot, including Engram tables, vision weights, all MTP stages, tokenizer files, and `exact-metadata/`. Do not use an exclude pattern that drops these files. Public model downloads do not require an API token; if your environment requires authentication, use `hf auth login` locally.

Checksum verification reads the complete download with a bounded buffer, so it takes substantial SSD read time. It does not load the model into memory. To verify a snapshot downloaded manually with `hf download`, run:

```sh
python deploy/deepseek/download_model.py --model-dir "$MODEL_DIR" --verify-only
```

## 3. Prepare exact metadata and configure

```sh
python deploy/deepseek/prepare_metadata.py \
  --model-dir "$MODEL_DIR" \
  --output "$METADATA_DIR"

./deploy/deepseek/service.sh configure \
  --model-path "$MODEL_DIR" \
  --metadata-proof "$METADATA_DIR/metadata-proof.json"
```

Preparation checks the verified download and the published metadata proof, reads bounded safetensors headers, and creates hardlinks for the 120 encoded metadata files. `METADATA_DIR` must be a **new directory outside the checkpoint on the same filesystem**. The downloaded checkpoint bytes are preserved, and no second overlay payload is copied. Preparation adds the local file identities needed by the runtime.

Configuration creates a private API key at `$OMLX_DS_STATE/api-key`, private settings, a model registry entry named `deepseek-v41-lsq`, and separate cache directories. The default endpoint is **127.0.0.1:8080**. To select another address or port, add `--host` and `--port` to `configure` before starting.

## 4. Start, stop, and restart

```sh
./deploy/deepseek/service.sh start
./deploy/deepseek/service.sh status
./deploy/deepseek/service.sh stop
./deploy/deepseek/service.sh restart
```

The scripts also work from another directory when called by their absolute path. Keep `OMLX_DS_STATE` set consistently if you use a non-default location.

`start` launches an independent detached process with its own PID, writes logs to `$OMLX_DS_STATE/logs/service.log`, and waits for model/runtime readiness. It refuses an occupied port or an unsafe memory situation. PID identity checks and lifecycle locking prevent duplicate launches and prevent signalling unrelated processes. `restart` waits for the owned process to exit before loading its replacement. A failed or timed-out start reports the log location.

The service continues after the terminal or chat closes. A PID-bound `caffeinate -i -w` assertion prevents idle sleep while this service is alive; it ends automatically with the service and does not change global macOS power settings. This manual launcher does not install a login item or automatic reboot/recovery service.

## 5. Use the API and dashboard

Open [http://127.0.0.1:8080/admin](http://127.0.0.1:8080/admin) and authenticate with your local key when prompted. The API is OpenAI-compatible:

```sh
export OMLX_DS_STATE="$HOME/.local/share/omlx-ds-256"
export OMLX_API_KEY="$(cat "$OMLX_DS_STATE/api-key")"

curl http://127.0.0.1:8080/v1/models \
  -H "Authorization: Bearer $OMLX_API_KEY"

curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Authorization: Bearer $OMLX_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v41-lsq","messages":[{"role":"user","content":"Explain exact prefix caching in two sentences."}],"max_tokens":256,"stream":true}'
```

## Memory policy and validation limits

The portable profile preserves the source deployment's **240 GiB process cap**, **238 GiB prefill target**, **16 GiB system budget**, **four request slots**, **8192-token prefill chunks**, **2 GiB exact-prefix RAM cache**, and **3 GiB prefill pool**. Admission depends on current physical headroom and reservations; four slots do not promise four arbitrary long contexts can run simultaneously. The configured 1,048,576-token context ceiling is not a validated workload size.

For acceptance runs, record a baseline and the maximum compressor physical occupancy throughout the run. Limits are **2 GiB peak compressor growth** and **100 MiB cumulative new swap-in plus swap-out**, with normal memory pressure and all process/runtime guards still enforced. Existing swap is part of the baseline. Do not increase memory ceilings to make a request fit.

The original runtime completed one cold **372,432-token multimodal request with 4,096 output tokens** on the machine above. That was a single-request validation; simultaneous long-context workloads are still limited by physical memory. Image-bearing long prefill can occupy a scheduler step for an extended period even while the HTTP server remains responsive.

This repository packaging was checked with CPU tests, source/patch checks, and simulated lifecycle tests. A fresh install of the portable package has **not** been qualified with model loading or inference. See [`validation.md`](docs/deepseek/validation.md) for the test scope, known numerical differences, and release limitations.

### Record memory and process exits

The optional recorder is an independent launchd job with a label unique to this state directory. It records host memory/pressure, process footprint, thermal information, and process-exit events without making inference requests:

```sh
./deploy/deepseek/recorder.sh start
./deploy/deepseek/recorder.sh status
./deploy/deepseek/recorder.sh stop
```

Recorder data is stored under `$OMLX_DS_STATE/logs/flight-recorder`. Start it before a long-context run so it captures the baseline. The bootstrap builds its small CPU telemetry helpers; no recorder is started automatically by installation.

For a bounded acceptance run, use a second terminal and start this observer before starting the model service:

```sh
"$OMLX_DS_STATE/venv/bin/python" deploy/deepseek/observe_omlx_cycle.py \
  --name validation-1 \
  --output-dir "$OMLX_DS_STATE/logs/cycles" \
  --allowed-compressor-growth-bytes 2147483648 \
  --allowed-global-swap-bytes 104857600 \
  --expect-final-stopped
```

After completing the workload and stopping the service with `service.sh stop`, end the observer with Ctrl-C, or create `$OMLX_DS_STATE/logs/cycles/validation-1.stop`. It writes samples and a summary, checks peak growth across the whole run, and verifies that the service has stopped. Use a new name for each run. The observer reports failures; it does not control or terminate the service. Runtime memory guards remain responsible for admission and execution.

## Source layout and provenance

| Location | Contents |
| --- | --- |
| `omlx/patches/deepseek_v41/` | DeepSeek model, quantization, native dispatch, Engram I/O and exact-cache implementation |
| `omlx/custom_kernels/` | Native attention and matrix-kernel source |
| `omlx/scheduler.py`, `omlx/request_memory_budget.py` | Request scheduling and memory reservation policy |
| `apps/omlx-web/` | Dashboard templates, translations and UI tests |
| `deploy/deepseek/` | Portable build, model preparation, guards and lifecycle controls |
| `docs/deepseek/source-manifest.json` | Imported source hashes and portability adaptations |

The complete imported customization set includes auxiliary and experimental paths; their presence does not enable them. The checked deployment profile selects the validated configuration. For general oMLX features, see the [upstream v0.7.3 README](https://github.com/jundot/omlx/blob/v0.7.3/README.md).

## License and acknowledgements

The oMLX project is distributed under [Apache-2.0](LICENSE). DeepSeek-derived code retains its [MIT license](omlx/patches/deepseek_v41/LICENSE), and MLX-derived native code retains its [Apple MIT notice](omlx/custom_kernels/attention/csrc/MLX_LICENSE.txt). See [third-party notices](docs/deepseek/THIRD_PARTY.md). Model weights are distributed separately under the terms in their Hugging Face repository.
