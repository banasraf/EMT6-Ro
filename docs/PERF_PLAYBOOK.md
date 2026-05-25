# EMT6-Ro performance playbook

How to measure, profile, and reason about throughput of the CUDA engine.
All commands assume you are at the repo root.

## 1. Build flags

Three CMake options drive the measurement scaffolding. All default OFF —
release builds without them are byte-equivalent to the reference source.

| Flag | When to turn on | Cost when on |
|------|-----------------|--------------|
| `-DEMT6RO_NVTX=ON` | Capturing a Nsight Systems / Nsight Compute trace. | Zero when no profiler is attached. |
| `-DEMT6RO_TIMING=ON` | You want a per-kernel ms breakdown from the Python API. | One `cudaEventSynchronize` per kernel per step. Negligible for typical runs (≤ 1 % of wall) but real. |
| `-DEMT6RO_INSTRUMENT=ON` | Counting repair / division / irradiation events. Unrelated to perf — for calibration runs. | A handful of `atomicAdd` calls in the hot kernels. |

Reference build for profiling work (Titan V, sm_70):

```bash
mkdir -p build && cd build
CC=gcc-11 CXX=g++-11 cmake -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=70 \
  -DEMT6RO_NVTX=ON -DEMT6RO_TIMING=ON ..
make -j$(nproc)
cd ..
ln -sf $PWD/build/python/emt6ro/simulation/backend.*.so python/emt6ro/simulation/
```

For RTX 5090 replace `70` with `120`. `native` works on CUDA ≥ 12.

## 2. Reference benchmark

`tools/perf_baseline.py` is the one shot you run every time you want a
number to commit to git. Always runs the canonical 10-day workload at the
largest batch_size that fits in device memory.

```bash
PYTHONPATH=$PWD/python python3 tools/perf_baseline.py
# or override batch:
PYTHONPATH=$PWD/python python3 tools/perf_baseline.py --batch-size 256
# fast sanity (5000 steps):
PYTHONPATH=$PWD/python python3 tools/perf_baseline.py --quick
```

Outputs land in `runs/perf-baseline-YYYY-MM-DD-<gpu>/`:

- `REPORT.md` — human-readable summary.
- `run.json` — machine-readable; ingest from notebooks.

Both capture git HEAD, GPU model, driver, CUDA runtime, CPU, batch size,
wall time per repeat, sims/sec, and per-kernel ms breakdown (zero unless
built with `EMT6RO_TIMING=ON`).

## 3. Nsight Systems timeline

For seeing kernel ordering, gaps, host-side overhead, and verifying NVTX
ranges fire as expected.

```bash
bash tools/profile_nsys.sh                 # quick (5000 steps)
STEPS=20000 bash tools/profile_nsys.sh     # longer trace
```

Open the resulting `.nsys-rep` in `nsys-ui`. Look for:

- **NVTX rows**: `step`, `findOccupied`, `updateROIs`, `diffuse`,
  `simulateCells`, `countLiving` ranges per iteration. Verify the
  cadence (`findOccupied` every 128 steps, `updateROIs` every 32).
- **CUDA HW row**: kernel occupancy, kernel duration, gaps. Gaps
  between kernels mean host-side overhead is dominant — that's the
  signal to look at launch counts or fusion opportunities.
- **Memory row**: `cudaMemcpyAsync` activity. If you see frequent
  host↔device copies, the result-collection path is hitting them every
  call instead of staying device-resident.

Cap trace size: a 144 000-step run produces a multi-GB trace that the
GUI struggles to open. Use `--quick` (5 000 steps) or `STEPS=20000` for
captures.

## 4. Nsight Compute drill-down

Per-kernel metrics: occupancy, memory throughput, warp execution
efficiency, branch divergence, register pressure, shared-memory bank
conflicts.

```bash
bash tools/profile_ncu.sh                          # default 5 launches per kernel
SKIP=500 COUNT=10 bash tools/profile_ncu.sh        # later launches, more samples
```

Profiles each kernel separately (cellSimulation, diffusion, findROIs,
findOccupied, countLiving) into its own `.ncu-rep`. NCU replays launches
to gather metrics from the `full` section set so wall time is several
minutes even at small batch.

### First-look metrics

For each kernel, eyeball in `ncu-ui`:

| Section | Field | Healthy on Titan V / RTX 5090 |
|---------|-------|-------------------------------|
| **Occupancy** | Theoretical / Achieved | > 50 % ; if low, look at register / shmem pressure |
| **Memory Workload Analysis** | Compute (SM) Throughput vs. Memory Throughput | Want one of them > 60 %. Both < 60 % = latency-bound. |
| **Memory Workload Analysis** | L1/TEX Hit Rate | Per-kernel; > 80 % is good for these access patterns (cells touch their 8 neighbours). |
| **Scheduler Statistics** | Eligible Warps Per Cycle | < 1 = latency-bound; > 2 = healthy |
| **Warp State Statistics** | Long Scoreboard / LG Throttle | If > 30 % stall reason: memory-bound |
| **Source Counters** | Branch Efficiency | < 85 % = significant divergence to investigate |
| **Launch Statistics** | Registers Per Thread | If 64+ and occupancy < 50 %, register pressure is the bottleneck |

## 5. Interpreting `REPORT.md`

The Markdown report includes a per-kernel table. Sample heuristic:

- **`diffuse` ≫ everything else**: diffusion stencil dominates. Likely
  candidates: shared-memory tile reuse, reducing the per-step substep
  count, persistent kernel that keeps state in shmem across steps.
- **`simulateCells` dominant**: cell-state update is on the hot path.
  Candidates: SoA migration (currently AoS — `src/emt6ro/site/site.h`),
  reduced divergence in the cell-cycle switch.
- **`findOccupied` / `updateROIs` material**: these run every 128 / 32
  steps. If their amortised share is significant, either (a) cadence is
  wrong or (b) the reduction kernel is launch-overhead-bound.
- **Gap between summed kernel ms and wall**: host-side overhead. Each
  `step()` makes 1–4 kernel launches per iteration; at 144 000 steps the
  launch latency floor matters. Look at CUDA Graphs / persistent kernels.

## 6. Reference numbers (placeholder)

Fill in after first baseline run:

| Date | Git | GPU | batch | wall (10d) | sims/sec | diffuse share | cells share |
|------|-----|-----|-------|------------|----------|---------------|-------------|
| TBD  | TBD | Titan V | TBD | TBD | TBD | TBD | TBD |

## 7. Gotchas

- **`Experiment(...)` positional args**: always pass `simulation_steps`
  and `protocol_resolution` as kwargs. Positional silently lands them
  in the wrong slots (the existing `tools/run_protocols_gpu.py` and
  `tools/perf_baseline.py` both use kwargs).
- **CUDA sandbox**: this dev box blocks `/dev/nvidia*` by default. Any
  CUDA-bound command (Python that calls `Experiment`, `nsys`, `ncu`)
  needs `dangerouslyDisableSandbox: true` in the Claude tool call, or
  just run it from a normal shell.
- **`EMT6RO_TIMING` cost is real**: the per-kernel `cudaEventSynchronize`
  serialises each kernel against the host. Don't use TIMING builds for
  throughput numbers you intend to publish; use them only for the
  per-kernel breakdown. The harness reports both wall time and per-kernel
  ms so you can see the overhead.
- **Default-flag build is canonical**: report headline numbers from
  builds WITHOUT `-DEMT6RO_TIMING=ON`. Use the timing build to attribute
  cost across kernels.
