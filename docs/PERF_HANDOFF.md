# EMT6-Ro perf work — handoff

Landing page for the next session. If you're an agent or a human picking
this up on a new machine (RTX 5090), read this first.

## TL;DR

- **Goal**: increase throughput of 10-day (144 000-step) EMT6-Ro CUDA simulations.
- **Status**: measurement scaffolding landed + first baseline captured on
  Titan V + 29 candidate diffusion-kernel optimisations enumerated.
  No optimisations applied yet.
- **Next step**: re-baseline on RTX 5090, then run the 29 diffusion tuning
  experiments via the Stage-A → Stage-B funnel described in
  `docs/PERF_EXPERIMENT_PLAYBOOK.md`.

## Where everything lives

| File | Contents |
|------|----------|
| `docs/PERF_HANDOFF.md` | This document. |
| `docs/PERF_PLAYBOOK.md` | How to build, run benchmarks, run nsys / ncu. |
| `docs/NCU_FINDINGS.md` | Titan V NCU drill-down with interpreted bottlenecks. |
| `docs/PERF_EXPERIMENT_PLAYBOOK.md` | Worktree fan-out + Stage A/B benchmark funnel. |
| `runs/perf-baseline-2026-05-20-titanv/REPORT.md` | First baseline numbers (Titan V). |
| `runs/perf-baseline-2026-05-20-titanv/run.json` | Machine-readable baseline. |
| `runs/perf-baseline-2026-05-20-titanv/OPTIMIZATION_INVENTORY.md` | First-pass agent inventory — 29 ideas across all kernels. Less specific. |
| `runs/perf-baseline-2026-05-20-titanv/DIFFUSION_TUNING_EXPERIMENTS.md` | Diffusion-kernel-specific — 29 experiments with concrete code edits. **Primary backlog for the next session.** |
| `runs/perf-baseline-2026-05-20-titanv/ncu_*.ncu-rep` | NCU profile reports — open with `ncu-ui` on macOS host install. |
| `tools/perf_baseline.py` | The harness. Autosizes batch, runs canonical workload, emits REPORT.md + run.json. |
| `tools/profile_nsys.sh` | Nsight Systems wrapper (timeline + NVTX). |
| `tools/profile_ncu.sh` | Nsight Compute wrapper (per-kernel metrics; needs sudo + counter perms). |

## Build flags introduced this session (all default OFF — byte-equivalent default build)

| Flag | Purpose |
|------|---------|
| `-DEMT6RO_NVTX=ON` | NVTX ranges around per-step kernel launches. Free when no profiler attached; needed for clean nsys timelines. |
| `-DEMT6RO_TIMING=ON` | Per-kernel CUDA-event timers; exposed via `Experiment.get_kernel_timers()`. Adds one `cudaEventSynchronize` per kernel per step — overhead is absorbed by kernel wait at saturating batch (verified <0.1% on Titan V baseline). |
| `-DEMT6RO_INSTRUMENT=ON` | (Pre-existing.) Per-event diagnostic counters / histograms. Unrelated to perf. |

Reference build for profiling / tuning work:

```bash
mkdir -p build && cd build
CC=gcc-11 CXX=g++-11 cmake -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DEMT6RO_NVTX=ON -DEMT6RO_TIMING=ON ..
make -j$(nproc)
cd ..
ln -sf $PWD/build/python/emt6ro/simulation/backend.*.so python/emt6ro/simulation/
```

(`sm_120` for RTX 5090; use `70` on Titan V, `native` if CUDA toolkit ≥ 12.5
supports it for the host GPU.)

## Headline baseline (Titan V, batch 30 750, 144 000 steps, 1 repeat)

| metric | value |
|--------|-------|
| Wall | 4459.65 s (74.3 min) |
| Throughput | 6.9 sims/sec |
| Diffuse | 68.1 % of time |
| simulateCells | 30.7 % |
| updateROIs | 1.0 % |
| findOccupied | 0.2 % |
| countLiving | 0.0 % |
| Host overhead | ~3 s of 4460 s — kernel-bound, NOT launch-bound |

**Diffusion is the prime target.** Full numbers in
`runs/perf-baseline-2026-05-20-titanv/REPORT.md`.

## Why diffusion was a surprise

The first optimisation-inventory agent assumed memory-boundness from the
AoS Site layout (Site is 56 B; diffusion reads only the 12-B `substrates`
field per cell). NCU **inverted** that hypothesis:

- DRAM throughput only **15 %** — global memory is not the bottleneck.
- L1/TEX throughput **53 %** — shmem reuse from the substep loop is
  already excellent (the kernel pulls ROI into shmem and iterates inside).
- Compute and Memory both **<40 %** simultaneously — classic
  **latency-bound** signature.
- Scheduler **no-eligible 53 %** of cycles. Active warps 7.6/16 → only
  **1 block per SM** because ~41 KB shmem/block blocks the second block.

So the AoS→SoA Site split (which the first agent recommended as a top
lever) **does not help diffusion**. It might still help `simulateCells` —
needs its own NCU pass.

The right diffusion levers (from `DIFFUSION_TUNING_EXPERIMENTS.md`):

1. Increase **occupancy** by reducing shmem footprint / register pressure.
2. Cut **substep barrier waits** via re-rolled loops, removed syncthreads.
3. Reduce **divergence** in the border-mask branch (~25 % lane waste).

29 specific experiments are enumerated in
`DIFFUSION_TUNING_EXPERIMENTS.md` — each is a semantics-preserving change
(no stencil swap, no substep-count reduction, no SoA layout — those are
separate validated changes tracked elsewhere).

## What to do first on the new machine

1. **Sanity rebuild** with `-DCMAKE_CUDA_ARCHITECTURES=120` (RTX 5090).
   Confirm the harness loads and produces non-zero kernel timers.
2. **Fresh baseline run** on RTX 5090:
   ```bash
   PYTHONPATH=$PWD/python python3 tools/perf_baseline.py --repeats 3
   ```
   Outputs `runs/perf-baseline-<date>-rtx5090/REPORT.md`. **Use this as
   the reference baseline for all subsequent comparisons**, not the
   Titan V numbers.
3. **Fresh NCU drill-down** on diffusion + simulateCells:
   - Fix counter permissions: `options nvidia
     NVreg_RestrictProfilingToAdminUsers=0` in
     `/etc/modprobe.d/nvidia-profiling.conf`, then `sudo
     update-initramfs -u && sudo reboot`. (Less hassle than the per-call
     `sudo ncu` we used on Titan V.)
   - `bash tools/profile_ncu.sh`
   - Compare the Blackwell numbers to the Titan V picture in
     `docs/NCU_FINDINGS.md`. The bottleneck might shift (e.g. larger L2,
     different shmem-config caps the occupancy story differently).
4. **Run the experiment funnel** as described in
   `docs/PERF_EXPERIMENT_PLAYBOOK.md`. One worktree per experiment,
   parallel implementation, **serial** benchmarking.

## Methodology rules

- **Always use kwargs for `simulation_steps` and `protocol_resolution`**
  in `Experiment(...)`. Positional args silently land in the wrong slots
  (the harness already uses kwargs; preserve this pattern in any new
  scripts).
- **Headline performance numbers come from `-DEMT6RO_TIMING=OFF` builds**.
  The TIMING build is correct for ranking experiments against each other
  (everyone pays the same host-sync overhead) but wrong for absolute
  claims you'd put in a paper / a commit message.
- **Reference baseline must be on the same machine** as the experiments.
  Cross-machine comparisons (Titan V vs RTX 5090) are not valid for
  judging an optimisation; they're only useful as scaling context.
- **Validate semantics for any non-trivial code change**: a small KS test
  (3 tumors × 5 protocols × 30 reps, final cell counts, |Cohen's d| < 0.2,
  KS p > 0.001) before claiming a perf win.

## Memory / palace

Prior-session context is stored in MemPalace under `wing=emt6ro` —
useful drawers:

- `emt6ro/architecture` → "EMT6-Ro perf benchmarking + profiling
  scaffolding"
- `emt6ro/benchmarks` → "EMT6-Ro first perf baseline" and
  "EMT6-Ro NCU drill-down — Titan V"
- `emt6ro/decisions` → "EMT6-Ro perf scope decisions"

OpenCode doesn't share Claude Code's MemPalace, so these are advisory
notes — the canonical source is the in-repo Markdown files referenced above.
