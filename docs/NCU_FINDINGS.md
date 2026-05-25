# NCU drill-down — Titan V baseline (2026-05-20)

Captured profiles + the interpretation that drove the diffusion-kernel
optimisation backlog. Re-do this on RTX 5090 before relying on the
ranking — bottlenecks can shift with arch.

## Profile run parameters

- Tool: `ncu --set full --launch-skip 20 --launch-count 3 --kernel-name regex:<K>`
- Build: `-DEMT6RO_NVTX=ON -DEMT6RO_TIMING=ON`, Release, sm_70
- Workload: 5000 steps × 120 sims (batch must be ≥ 10 × n_tumors)
- Raw artefacts: `runs/perf-baseline-2026-05-20-titanv/ncu_*.ncu-rep`
- Driver / NCU: 555.42.06 / 2024.2.1.0
- Note: needs `sudo` until `NVreg_RestrictProfilingToAdminUsers=0` is set
  system-wide. See `docs/PERF_HANDOFF.md`.

## diffusionKernel — LATENCY-BOUND (the prime target)

### GPU Speed-Of-Light

| metric | value | meaning |
|--------|-------|---------|
| Compute (SM) Throughput | **38.11 %** | moderate, below 40 % |
| Memory Throughput | **39.17 %** | moderate, below 40 % |
| DRAM Throughput | **15.29 %** | LOW — global memory NOT the bottleneck |
| L1 / TEX Throughput | **53.02 %** | highest single utilisation — heavy shmem traffic in the substep loop |
| L2 Throughput | 28.16 % | moderate |
| SM Active / Elapsed | 89 155 / 122 850 = 72.5 % | SMs stalled 27.5 % of cycles |
| Duration | 102.14 µs per launch | |

Both Compute and Memory below 40 % at the same time → NCU's textbook
**latency-bound** classification.

### Warp State

| metric | value | meaning |
|--------|-------|---------|
| Warp Cycles Per Issued Instruction | **16.28** | vs ideal ~1-2; HEAVY latency between instructions |
| Avg Active Threads / Warp | **23.67 / 32 = 74 %** | ~8 lanes idle per warp ⇒ ~25 % lane waste |
| Avg Not-Predicated-Off | 22.70 / 32 = 71 % | ~29 % of lanes masked off each instruction |

### Scheduler Statistics

| metric | value | meaning |
|--------|-------|---------|
| Active Warps Per Scheduler | **7.61** | vs Volta max 16 → ~1 block/SM resident |
| Eligible Warps Per Scheduler | **1.60** | just above latency-bound threshold of 1.0 |
| **No Eligible** | **53.23 %** | half the cycles, scheduler had nothing to do |
| Issued Warp Per Scheduler | 0.47 | 47 % of issue capacity |

### Root-cause synthesis

- ~41 KB shmem/block (bordered ROI tile `(55+4)² × sizeof(Substrates=3×float)`)
  exceeds Volta's ~48 KB soft cap for 2-blocks-per-SM → **1 block per SM**.
- 16-cycle warp-stall × 53 % no-eligible-cycles → scheduler is starved.
  Either `__syncthreads()` barrier waits, shmem-load latency, or RAW
  dependency chains in the substep loop. Need stall-reason breakdown
  (collected but not yet pasted into this doc).
- ~25 % lane waste from divergence — border-mask three-way branch
  (`src/emt6ro/diffusion/diffusion.cu:154-161`), variable ROI size early
  in the sim, `GRID_FOR` macro trip-count variance.

### What this implies for optimisations

The first optimisation-inventory agent put AoS → SoA `Site` split as a top
diffusion lever (3× bandwidth saving for the 12-B `substrates` field
loaded from a 56-B Site). NCU **inverts** this: DRAM at 15 % means
bandwidth is not the constraint here. SoA split is **not** the right
diffusion lever (may still help `simulateCells` — pending).

The right diffusion levers:

1. **Reduce shmem footprint** → 2-3 blocks/SM possible → directly
   attacks the 53 % no-eligible. Examples: per-substrate kernel split
   (separate validation needed since this is a kernel-boundary change),
   smaller working set per block, register-resident state.
2. **Cut substep barrier waits** → re-rolled loops, removed syncthreads
   where warp-level guarantees suffice.
3. **Reduce divergence** → predicated border-mask, more uniform site
   distribution.

29 specific semantics-preserving experiments derived from this analysis
are in `runs/perf-baseline-2026-05-20-titanv/DIFFUSION_TUNING_EXPERIMENTS.md`.

## findOccupied — MEMORY-BOUND, no locality (small kernel, low priority)

Profiled as a methodology check; only 0.2 % of total wall, so don't
optimise alone.

| metric | value |
|--------|-------|
| Compute Throughput | 1.44 % |
| Memory Throughput | 47.97 % |
| DRAM Throughput | 47.97 % (= Memory — all in HBM) |
| L1 / TEX | 10.27 % |
| L2 | 15.14 % |

Reads 56-B `Site` for one byte of `state`. Useful-byte rate = 1.8 %.
Smoking gun for the AoS waste pattern — relevant to `simulateCells` and
to a future Site SoA migration, but findOccupied itself isn't worth
isolated work.

## simulateCells — NOT YET PROFILED

Needs an NCU pass on the new machine. Hypothesis (to be confirmed):

- Heavy AoS Site traffic per cell (`cell` field is ~40 B, `state` 1 B,
  `substrates` 12 B — all read per step per occupied cell).
- Likely memory-bound or memory-coalescing-bound, unlike diffusion.
- 30.7 % of total time — second priority after diffusion.

If the NCU pass confirms memory-bound: SoA Site split likely helps here.

## What to capture next NCU pass

Sections we did not record fully on Titan V (re-do on RTX 5090):

1. **Source Counters → Warp Stall Reasons** — confirms whether the
   16-cycle stall is `Stall Barrier` (syncthreads), `Stall Long
   Scoreboard` (global memory), `Stall Short Scoreboard` (shmem), or
   `Stall Wait` (fixed-latency). This determines which of the 29
   experiments is the highest-priority first cut.
2. **Launch Statistics → Static Shared Memory Per Block** — confirms the
   ~41 KB figure (estimated from `(55+4)² × 12`).
3. **Occupancy → Theoretical Active Blocks Per SM Limiter** — should say
   "Shared Memory" if the hypothesis is right; if "Registers" then
   register-pressure mitigation jumps ahead in the priority list.
4. **Memory Workload Analysis → Shared Memory → Bank Conflicts** — if
   non-zero, swizzle / padding experiments (#4 in the tuning doc)
   move up the priority list.
5. **simulateCells** — full SOL + warp + scheduler stats, same as
   diffusion.
