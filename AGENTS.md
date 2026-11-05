# Repository operating requirements

This checkout maintains source code. Never start inference, load model weights,
contact an existing deployment, or stop another service as part of ordinary
packaging, documentation, or CPU-test work. Use an explicitly authorized runtime
validation window for live tests. Keep runtime state and models outside this
repository.

For authorized deployments:

- Run oMLX as its own detached operating-system process with its own PID, logs,
  and lifecycle, independent of any chat, application, terminal, or SSH session.
  Management clients use APIs and process controls; they never load another model.
- Preserve physical-memory guards and admission budgets. Account for model
  weights, prefix/KV caches, temporary allocations, other processes, and the OS.
  Do not weaken guards to load a model or force a benchmark to pass.
- Check physical memory, memory pressure, compressor occupancy, and swap before
  and throughout heavyweight validation. Record a starting baseline. Normal
  operation must fit physical memory without sustained compression or paging.
- Full-cycle acceptance permits at most 2 GiB (2147483648 bytes) of peak growth
  in the compressor's actual physical occupancy, and at most 100 MiB (104857600
  bytes) of cumulative new swap-in plus swap-out. Existing swap is baseline.
  Logical input bytes sent to compression are diagnostics, not the acceptance
  limit. Normal memory pressure and footprint/runtime guards remain enforced.
  Use `deploy/deepseek/observe_omlx_cycle.py` with both explicit allowances.
- Stop competing heavyweight runtimes only when authorized, and confirm memory
  is released before loading a replacement. SSD prefix-cache persistence is
  distinct from operating-system swap.
- Preserve the public executable start/stop/restart shell entrypoints. They must
  work from any directory, avoid duplicate launches, verify readiness and PID
  identity, report logs and failures, lock lifecycle operations, and wait for
  termination before replacement. Stop must never target an unrelated process.
- For lifecycle changes, use CPU/mocked tests during source maintenance. During
  an authorized live validation, exercise the public entrypoints and confirm
  that termination releases model memory.
