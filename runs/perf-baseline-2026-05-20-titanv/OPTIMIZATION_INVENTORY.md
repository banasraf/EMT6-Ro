# EMT6-Ro CUDA optimization inventory

Baseline (Titan V, batch 40, 5000 steps, EMT6RO_TIMING=ON):
`diffuse` 66.5% (242 ms), `simulateCells` 31.8% (116 ms), `updateROIs` 1.0%, `findOccupied` 0.7%, `countLiving` 0%. Target = RTX 5090 (sm_120, 32 GB). Per-step kernel launches = 4 → 576 k per 144 k-step run.

Items grouped by area, ordered by impact estimate within area.

---

## 1. Diffusion kernel (66.5% — top priority)

### 1.1 Drop the 9-point stencil's diagonal arm — or relegate it to a final corrector

**Where:** `src/emt6ro/diffusion/diffusion.cu:122-133` (stencil), `:165-172` (substep loop).

**Now:** Each substep does 4 axial + 4 diagonal shmem reads, diagonals weighted by `M_SQRT1_2`.

```cpp
// diffusion.cu:122-132
Substrates result = lattice(r - 1, c) + lattice(r, c - 1) +
                    lattice(r + 1, c) + lattice(r, c + 1);
result += (lattice(r - 1, c - 1) + lattice(r - 1, c + 1)  +
           lattice(r + 1, c - 1) + lattice(r + 1, c + 1)) *
          M_SQRT1_2;
result -= lattice(r, c) * f;
result *= coeffs;
```

**Change:** 5-point Laplacian for the 24 inner substeps (`time_step/diffusion_time_step = 6/0.25 = 24`), optionally one 9-point corrector at the end. Recoeff to preserve steady-state rate.

**Impact:** Large — diagonals are ~44% of shmem traffic per substep. Likely 25–35% on diffuse → ~17–23% overall.

**Risk:** Moderate; re-validate vs C++ ground-truth and re-run τ-calibration.

**Deps:** None.

---

### 1.2 Reduce substep count — 24 explicit Jacobi substeps look over-resolved

**Where:** `simulation.cu:241` passes `params.time_step/params.diffusion_params.time_step = 24`; loop at `diffusion.cu:165-172`.

**Now:** 23 in-shmem substeps + 1 write-back. CFL with D≤1.34, h=1 admits Δt ≤ ~0.125 for a 5-pt; the 9-pt's `4+2√2` denominator is slightly more permissive.

**Change:** (a) Empirically try 12 / 8 substeps and check accuracy. (b) Red-black Gauss-Seidel — halves substep count for same convergence. (c) ADI / Thomas (53×53 tridiag is cheap, params constant across batch) — collapse to 1 substep.

**Impact:** Linear in substep count. 24→12 → diffuse halved → ~33% overall.

**Risk:** Moderate — stability + ground-truth validation. ADI is more work.

**Deps:** Order with 1.1: changing stencil shape changes CFL threshold.

---

### 1.3 Per-substrate kernel split (or SoA shmem planes)

**Where:** `diffusion.cu:135-191`. Shmem footprint per block = `(ROI.h+4)(ROI.w+4) × 12 B`. At full-grown ROI ≈ 51×51 → ~36 KB, limiting Titan V to ~2 blocks/SM.

**Now:** AoS `Substrates {cho, ox, gi}` in shmem; substrates iterate in lock-step but never interact within the loop.

**Change:** (a) Run one kernel per substrate (12 KB shmem → 3× occupancy, 3 sequential launches). (b) Or three separate `float` planes in shmem within one kernel — same data, less padding waste.

**Impact:** Large but uncertain — 1.5–2× on diffuse if shmem-occupancy bound. Confirm with ncu `smsp__warp_issue_stalled_short_scoreboard`.

**Risk:** Moderate — pipe diffusion contract through.

**Deps:** Synergistic with 4.1.

---

### 1.4 Pick block dim from ROI; small ROI shouldn't get 32×32

**Where:** `diffusion.cu:13-14, 197`. `dim3(32,32)` fixed; `kSitesPerThread=4` (4× 12-B Substrates + 4× 4-B Coords = 64 B local per thread).

**Now:** Block sweeps ROI via `GRID_FOR`; threads outside ROI still hit barriers.

**Change:** 16×16 block when ROI small (first ~30 k steps); 32×32 when saturated. Or just default to 16×16 throughout and measure.

**Impact:** Moderate for growth phase, small at saturation. ~10–15% on diffuse total.

**Risk:** Low.

**Deps:** None.

---

### 1.5 Recompute `border_masks` only when ROI changes

**Where:** `diffusion.cu:65-100`, called every 32 steps via `simulation.cu:211-217`.

**Now:** Border mask rewritten unconditionally on every `updateROIs`.

**Change:** Cache previous ROI; skip when unchanged. Or eliminate the mask entirely and use an inline radial cutoff in the diffusion kernel.

**Impact:** Small — updateROIs is 1% total.

**Risk / Deps:** Low / none.

---

### 1.6 Shmem bank conflicts in the 9/5-pt stencil

**Where:** `diffusion.cu:135-191`. 12-B Substrates × stride 55 floats almost certainly conflicts on the 32-bank shmem.

**Change:** Pad each row to 33 floats per plane (after 1.3 splits planes).

**Impact:** Small to moderate — confirm with ncu `l1tex__data_bank_conflicts_pipe_lsu_mem_shared`. Probably <10% of diffuse.

**Deps:** Item 1.3.

---

## 2. Cell simulation kernel (31.8%)

### 2.1 Tighten thread count 512 → 128, raise occupancy

**Where:** `simulation.h:141` (`simulate_num_threads = 512`), kernel at `simulation.cu:93-137`.

**Now:** 512 threads/block. With `curandStateXORWOW` = 48 B/thread → 24 KB/block of RNG state alone (lives in lmem/global with caching cost). Each thread handles ≤2 of up to 1024 occupied cells.

```cpp
// simulation.h:141
int simulate_num_threads = 512;
// simulation.cu:99-103
uint8_t vacant_neighbours[4];
curandState_t *rand_state =
    rand_states + blockDim.x * blockIdx.x + threadIdx.x;
```

**Change:** Drop to 128 threads/block; each loops over ~8 cells. RNG footprint /4. Validate `block_reduce` at the smaller block (it's templated, should work).

**Impact:** Moderate to large — if occupancy-bound on per-block state, 2–3× occupancy → 20–40% on simulateCells → 6–13% overall.

**Risk:** Low complexity, moderate testing.

**Deps:** Pairs with 2.2.

---

### 2.2 Swap `curandStateXORWOW` (48 B) → `curandStatePhilox4_32_10` (16 B)

**Where:** `random-engine.h:35`, `random-engine.cu:8-19`, `CuRandEngine`.

**Now:** XORWOW per thread; ~750 MB of RNG state at batch 30 750, contending with L2.

**Change:** Philox 4-32-10 — counter-based (deterministic re-runs possible), 3× smaller, ~1.5× faster per uniform on modern GPUs.

**Impact:** Moderate — 5–15% on simulateCells; meaningful L2 pressure relief.

**Risk:** Low.

**Deps:** None.

---

### 2.3 Replace serial prefix-sum in `findOccupied` with `cub::BlockScan`

**Where:** `simulation.cu:69-91`.

**Now:** 32-thread kernel with `for (int i = 0; i < threadIdx.x; ++i) acc += shmem[i];` — quadratic on threads.

**Change:** `cub::BlockScan` or warp-shuffle scan; useful if the kernel grows beyond 32 threads (necessary if we drop the static 1024 cap).

**Impact:** Small — findOccupied is 0.7%.

**Risk / Deps:** Low / none.

---

### 2.4 Allow multiple non-conflicting divisions per step

**Where:** `simulation.cu:97-136`. `block_reduce(...)` collapses all division candidates to one per block per step.

**Now:** One division/block/step; congestion in growth phase.

**Change:** 2-colour (red/black) neighbour scheme + `atomicCAS` on `state` to claim child coords. Multiple non-conflicting divisions per step.

**Impact:** Small to moderate in growth phase (first ~30 k steps).

**Risk:** Moderate — touches division correctness. Note: tumor-ca CPU already divides ALL ready cells/step (palace memo) — fixing this is partly a correctness alignment.

**Deps:** None.

---

### 2.5 Branchy `Site::step` — sort the occupied stack by (mode, phase)

**Where:** `site.h:62-82`, `cell.cu:33-82`.

**Now:** Per-thread phase/mode/threshold branches → warp divergence.

**Change:** Sort the occupied stack by (mode, phase) inside `findOccupied`; threads in a warp see same-state cells.

**Impact:** Small — 5–10% on simulateCells.

**Risk / Deps:** Low / none.

---

## 3. Per-step host loop / launch overhead

### 3.1 CUDA Graphs — capture the step, replay 144 k times

**Where:** `simulation.cu:199-237`. 4 launches × ~5 µs ≈ 20 µs of pure launch latency/step → ~3 s floor.

**Now:** Plain `kernel<<<...>>>` per step. Non-uniform cadence (findOccupied/128, updateROIs/32, diffuse+simulate/1).

**Change:** Capture three graph variants matching the cadence:
- `step_full` (findOccupied + ROIs + diffuse + sim) — 1/128
- `step_roi`  (ROIs + diffuse + sim) — 3/128
- `step_inner` (diffuse + sim) — 124/128

Replay via `cudaGraphLaunch`. ~1 µs/replay ≪ 4 raw launches.

**Impact:** Moderate at small batch (kills launch-overhead floor), small at large batch where work dominates. Cheap insurance.

**Risk:** Moderate — `step` is a kernel arg today; either patch via `cudaGraphExecKernelNodeSetParams` or promote `step` to a device counter (item 3.3).

**Deps:** Item 3.3 makes graph replay trivial.

---

### 3.2 Merge `findOccupied` + `updateROIs` and/or maintain stack incrementally

**Where:** `simulation.cu:201-217`. Both scan the lattice for occupied cells at different cadences.

**Change:** (a) Run both every 32 steps in one fused pass. (b) Better: maintain the occupied stack incrementally — `cellSimulationKernel` already knows births/deaths, just queue them.

**Impact:** Small — these are 1.7% combined. Incremental version reduces staleness too.

**Risk:** Low for fusing, moderate for incremental (concurrent push/pop).

---

### 3.3 Promote `step_` counter to device memory

**Where:** `simulation.cu:235, 249` — `step_` is host-side, passed by value.

**Change:** Single-element `device::buffer<uint32_t> step_counter`. Cell kernel increments. Whole 144 k-step loop = one graph replay, no host interaction.

**Impact:** Small alone; enables clean 3.1.

**Deps:** Prerequisite for 3.1's cleanest form.

---

## 4. Data layout / memory traffic

### 4.1 Split `Site` AoS → SoA (substrates / cell / state)

**Where:** `src/emt6ro/site/site.h:34-43`. `Site = Substrates (12) + Cell (~40) + State (1) + pad ≈ 56 B`.

**Now:** Diffusion loads the full 56-B Site to read 12-B substrates — 75% bandwidth waste. Same for cell sim reading cell+state.

```cpp
// site.h:41-43
Substrates substrates;
Cell cell;
State state;
```

**Change:** Three parallel buffers per sim:
- `substrates_cho/ox/gi[]` (separate planes or a single `float3[]`),
- `cells[]` (~40 B each),
- `states[]` (1 B each).

Diffusion → 12-B-per-cell loads (3× saved). simulateCells touches all three but writes substrates separately.

**Impact:** Large for diffusion (bandwidth-bound stencil) — possibly 1.5–2× on diffuse → 30–35% overall.

**Risk:** High — touches every kernel + state.cc + Python accessors + tests. Mechanical but large surface.

**Deps:** None. Pairs with 1.3 and 4.2.

---

### 4.2 Pad lattice row stride 53 → 64

**Where:** Dims at `simulation.cu:152` = `(53, 53)`. Non-power-of-2 stride → misaligned warp accesses.

**Change:** Pad to 64 (or 56). Skip pad cells.

**Impact:** Small without 4.1, moderate with.

**Risk / Deps:** Low / 4.1.

---

### 4.3 `occupied` stack capped at 1024 — verify it's never exceeded

**Where:** `simulation.cu:73-76, 85, 159` — `device::buffer<uint32_t> occupied(batch_size * 1024)`. Grid has 2809 interior cells; a saturated tumor could overflow.

**Change:** Add a runtime assertion; if real, bump cap to 2048 or 4096.

**Impact:** Correctness, not perf — but listed because silent overflow would corrupt late-phase populations.

**Risk / Deps:** Trivial / none. Do this regardless.

---

## 5. Batch / parallelism shape

### 5.1 Multiple sims per block

**Where:** `simulation.cu:197, 247` (`<<<batch_size, ...>>>`).

**Now:** 1 sim = 1 block. Titan V has 80 SMs × ~2 resident blocks (shmem-limited) ≈ 160 sims in flight. Tail latency at huge batch.

**Change:** Pack N sims into `threadIdx.z`; per-sim shmem partitioning; replace `__syncthreads()` with cooperative groups or sim-scoped barriers.

**Impact:** Moderate — potentially 1.5×.

**Risk:** High — invasive rewrite.

**Deps:** Combines with 1.3, 2.1, 2.2 (smaller per-sim state).

---

### 5.2 Persistent kernel — one CTA per SM, full simulation inside

**Where:** Whole `Simulation::step()` loop.

**Change:** One kernel per sim that loops internally, cooperative_launch for cross-block sync, lattice resident in shmem across steps.

**Impact:** Largest possible — 20–40% in large-batch regime.

**Risk:** Very high — full rewrite, requires cooperative launches; shmem must fit lattice + state.

**Deps:** Try 3.1 first (cheaper, captures most launch-overhead win).

---

## 6. Numerical / algorithmic

ADI / red-black GS already covered in 1.2. CHO/OX/GI diffusion is uncoupled (no inter-substrate term in `diffusion.cu`), so per-substrate splits are clean. Cell-cycle progression is already O(1) — no opportunity worth pursuing.

---

## 7. Build / compile

### 7.1 Add explicit `-O3`, `--use_fast_math`, `__launch_bounds__`

**Where:** `CMakeLists.txt:1-93` — no explicit `-O3` for CUDA, no fast-math, no `__launch_bounds__` on hot kernels.

**Now:** Release defaults to nvcc's implicit `-O3` (which is fine), but no `--use_fast_math` — `expf` (cell.cu:96, cell.h:110) and `sqrtf` (diffusion.cu:34) hit the precise math.h paths.

**Change:**
```cmake
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-O3>)
option(EMT6RO_FAST_MATH "Enable nvcc --use_fast_math" OFF)
if(EMT6RO_FAST_MATH)
  add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)
endif()
```

Annotate kernels: `__launch_bounds__(128, 8)` on cellSimulationKernel (post-2.1), `__launch_bounds__(1024, 2)` on diffusionKernel.

**Impact:** Small to moderate — `-O3` (if not already) plus fast-math could be ~10%. `__launch_bounds__` helps register choice → occupancy.

**Risk:** Trivial. Fast-math affects τ-calibration sensitivity — gate behind a flag, re-calibrate.

**Deps:** None.

---

### 7.2 Test disabling `CUDA_SEPARABLE_COMPILATION` for the static lib

**Where:** `src/emt6ro/CMakeLists.txt:34`, `src/CMakeLists.txt:24`.

**Now:** `CUDA_SEPARABLE_COMPILATION ON` blocks some cross-TU inlining of `Site::step`, `tryRepair`, `diffusion_differential`.

**Change:** Try `OFF` on the `emt6ro` static lib; tests target may still need it.

**Impact:** Small to moderate — 5–15% on hot kernels via inlining.

**Risk:** Low if it builds; could break device-symbol resolution.

**Deps:** None.

---

## RECOMMENDED ORDER (impact-per-effort)

1. **Build flags + `__launch_bounds__` + (gated) fast-math** (7.1, 7.2) — 1 day, possibly ~10–15% overall. Cheapest concrete win.
2. **CUDA Graphs + device-side step counter** (3.1 + 3.3) — 2–3 days. Kills the ~3 s launch-overhead floor, works at any batch.
3. **Diffusion substep reduction** (1.2: try 12 substeps; then red-black GS) → then **9-pt → 5-pt** (1.1). Together possibly halve the 66% diffusion cost. Mandatory re-validation against C++ ground-truth.
4. **Site SoA migration** (4.1) + **per-substrate diffusion** (1.3) + **row padding** (4.2) — one combined layout pass. Big rewrite but unlocks bandwidth.
5. **simulateCells: 512 → 128 threads** (2.1) + **Philox RNG** (2.2). ~25–40% on simulateCells with low correctness risk.

Deferred / out of scope for a first sprint: persistent kernel (5.2), multiple sims per block (5.1), ADI implicit diffusion (6.1), division-bottleneck fix (2.4 — tangled with known CPU/GPU divergence; do alongside τ-calibration). Item 4.3 (occupied-stack cap) is correctness, not perf — do anyway.
