# Non-intrusive diffusion-kernel tuning experiments

Target: `emt6ro::diffusionKernel` at `src/emt6ro/diffusion/diffusion.cu:135-191`. Scope: changes that preserve bit-identical (or FP-reassociation-bound equivalent) output. Algorithmic changes (stencil swap, substep count, per-substrate split, SoA layout) are tracked separately and excluded here.

Baseline NCU (Titan V, batch 120, 5000 steps): Compute SOL 38.1 %, Mem SOL 39.2 %, DRAM 15.3 %, L1/TEX 53.0 %, warp cycles/issued 16.28, **no-eligible 53.2 %**, active warps/scheduler 7.61 (~1 block/SM), avg active threads/warp 23.67/32. Latency-bound, occupancy-limited by ~41 KB shmem/block.

---

## Memory / shmem-read reduction

### Experiment 1 - Lift the 9 stencil reads into named local registers
**Hypothesis**: At `:122-133` each `lattice(r±1,c±1)` is a separate `LDS` against a 12-B `Substrates`. With the `+= diff[i]` write-back in the previous substep, the compiler may fail to prove non-aliasing of `tmp_grid.data` across the 9 reads. Forcing 9 explicit named locals plus one compound expression reduces L1/TEX issue pressure (53 % SOL is the highest single metric).
**Change**: rewrite `diffusion_differential` (`diffusion.cu:122-133`) to `const Substrates n = lattice(r-1,c); const Substrates s = lattice(r+1,c); ...` then compose. ~15-line diff.
**How to measure**: `smsp__inst_executed_pipe_lsu` (LSU pipe instructions) should drop a few %; warp cycles/issued should ease slightly; `l1tex__t_requests_pipe_lsu_mem_shared` should be unchanged but cycles down.
**Risk**: Pure refactor; FP order preserved exactly. Same SASS likely if the compiler was already CSE'ing - worth verifying with `cuobjdump --dump-sass`.
**Validity**: Bit-identical substrate field after one step (`Experiment.state()[i].cho/ox/gi` SHA256 match).

### Experiment 2 - Cache the eight neighbour reads in registers across the substep loop body, reuse for the `+=` step
**Hypothesis**: The differential at `:167` reads the 9 neighbours; the apply at `:170` immediately re-reads `tmp_grid(sites[i].r, sites[i].c)` to add `diff[i]`. The 8 neighbours are *not* reused across substeps (every site mutates) but the center *is* read+written in the same substep. Stashing the center read avoids one `LDS.64` per site per substep (~21k×24 LDS per kernel).
**Change**: in the substep loop (`:165-172`), keep a register `Substrates center[kSitesPerThread]` filled during the differential and use `tmp_grid(sites[i].r, sites[i].c) = center[i] + diff[i]` in the apply. ~6 lines.
**How to measure**: `l1tex__t_sectors_pipe_lsu_mem_shared_op_ld` should drop by ~1/9 (~11 %). Watch warp-cycles/issued.
**Risk**: Trivial. FP-bit-identical.
**Validity**: SHA256 of `tmp_grid` after each substep equal to baseline (instrumented build).

### Experiment 3 - Convert struct-of-3-floats accesses to explicit per-field reads
**Hypothesis**: `Substrates += Substrates` (`substrates.h:14-19`) is three scalar `+=`. ptxas usually handles this but with `-dc` separable compilation enabled (`CMakeLists.txt:10`) cross-TU inlining can fail. Forcing the kernel to address `cho/ox/gi` as 3 separate `float` planes-in-AoS reads ensures the 12-B group goes as one `LDS.64 + LDS.32` rather than three `LDS.32`s.
**Change**: rewrite `diffusion_differential` to do CHO sum, OX sum, GI sum in unrolled fashion. ~25 lines.
**How to measure**: SASS inspection: count `LDS.64` vs `LDS.32` in the substep loop. `l1tex__t_bytes_pipe_lsu_mem_shared` unchanged but issue count down.
**Risk**: None (FP-bit-identical; same arithmetic, same order per field).
**Validity**: SHA256 of substrate fields equal.

### Experiment 4 - Pad shmem row stride from 55+4=59 to 64 (or use 60) to avoid 32-bank conflicts
**Hypothesis**: `tmp_grid` row stride is `(roi.w+4) * 3 floats` = 59 x 3 = 177 floats per row (AoS). Stride 177 mod 32 = 17, so "vertical" stencil reads in a warp likely hit conflicting banks. 53 % L1 SOL is consistent with bank conflicts beyond ideal traffic.
**Change**: in `batchDiffusion` (`diffusion.cu:193-200`) pass a padded width to the kernel; in `diffusionKernel` (`:143`) use `Dims(roi.dims.height+4, padded_width)` for `bordered_dims`. ~10 lines.
**How to measure**: `l1tex__data_bank_conflicts_pipe_lsu_mem_shared` - the *direct* bank-conflict counter. Expect drop; secondary expect L1/TEX SOL to drop below 50 %.
**Risk**: Trivial; just larger shmem footprint (worsens occupancy slightly - 41 KB to ~42 KB, still 1 block/SM on Volta). FP-bit-identical.
**Validity**: SHA256 substrates equal.

### Experiment 5 - Read border mask once into a per-thread bitmask register
**Hypothesis**: `b_mask` (`:148-149`) is read at load (`:156`); occupancy is implicit in `sites[]` membership. Pre-packing a `uint32_t` mask of the 4 owned sites is only useful if Experiment 8 fuses load with substep iteration.
**Change**: at load time build `uint32_t mask_bits`. ~5 lines.
**How to measure**: combination-only.
**Risk**: None standalone. **Validity**: SHA256.

### Experiment 6 - Cache `coeffs` and `f` in `__constant__` memory, drop `coeffs` from kernel args
**Hypothesis**: `coeffs` (`diffusion.cu:163`) is computed per-block from kernel args (`params.coeffs * params.time_step * HS / f`) and lives in a register; it's the same value for every block in a launch. Moving the underlying scalars to `__constant__` lets ptxas use `LDC.32` (broadcast through constant cache) rather than carrying through registers, freeing 1-2 registers and potentially helping occupancy. With reg pressure low this is marginal; with `__launch_bounds__` it might matter.
**Change**: declare `__constant__ Substrates d_coeffs;` plus `__constant__` HS/f, copy via `cudaMemcpyToSymbol` from `batchDiffusion`. ~20 lines.
**How to measure**: `launch__registers_per_thread` (should drop 1-2), watch `sm__warps_active.avg.pct_of_peak_sustained_active` for any uplift.
**Risk**: Marginal. FP-bit-identical.
**Validity**: SHA256.

---

## Synchronisation

### Experiment 7 - Drop second `__syncthreads()` at `:171` by re-rolling the loop
**Hypothesis**: Much of the 53.2 % no-eligible cycles is likely `Stall Barrier`. Substep loop has two syncs per iteration (`:168, :171`). Re-rolling so the loop body is `apply; sync; differential;` with a peeled first differential lets each substep need only one sync - ~50 % fewer barrier stalls.
**Change**: peel the first differential outside the loop; loop body becomes `apply; sync; differential;`. ~15 lines around `:165-172`.
**How to measure**: `smsp__warp_issue_stalled_barrier_per_warp_active` should drop ~50 %. Watch no-eligible % - expect ~10-15 pp drop.
**Risk**: Off-by-one trap on the final iteration - the post-loop block at `:173-184` already does one more differential+apply. Reorder must preserve the total of `steps` differential+apply pairs.
**Validity**: SHA256 of substrates after kernel return.

### Experiment 8 - Fuse load phase with first substep's differential
**Hypothesis**: The load loop (`:150-162`) ends with `__syncthreads()` (`:164`). After load each thread holds its centers in registers; could compute the first differential on cached values immediately. Marginal.
**Change**: stash newly-written value into a local in the load loop. ~20 lines.
**How to measure**: Total kernel duration drops by ~one barrier+one substep.
**Risk**: Neighbours of own sites may not yet be visible to *other threads*. Existing sync still required for cross-thread reads.
**Validity**: SHA256. Higher risk than 7 - skip unless 7 lands cleanly.

### Experiment 9 - Replace barrier with `__syncwarp` in single-warp case
**Hypothesis**: When ROI is tiny (~5x5), only ~32 threads do real work but `__syncthreads()` bars all 1024. Branch on `__ballot_sync(nsites>0)` and pick `__syncwarp` vs `__syncthreads`. Only relevant in startup (first ~30k steps).
**Change**: ~10 lines. **How to measure**: only visible at small ROI - profile separately.
**Risk**: Subtle - any thread that wrote shmem visible to *another* warp still needs the full barrier. **Validity**: SHA256.

---

## Compute / FP / register

### Experiment 10 - `#pragma unroll` on the `kSitesPerThread=4` site loops
**Hypothesis**: `for (i = 0; i < nsites; ++i)` at `:166, :169` has variable `nsites` (0..4). `#pragma unroll 4` plus a runtime mask would unroll the loop and predicate the tail. With nsites in `{0,4}` most blocks (steady state has near-full coverage by owned-sites) the predicated path is free. Removes loop-control overhead in the hot path.
**Change**: at `:166, :169` add `#pragma unroll`. ~2 lines.
**How to measure**: `smsp__inst_executed.sum` should drop a few %. Look for `ISETP` and `BRA` count reduction in SASS.
**Risk**: Slight register-pressure bump (4 `diff[]` and 4 site coords already in regs - should be a no-op). FP-bit-identical.
**Validity**: SHA256.

### Experiment 11 - Hoist `M_SQRT1_2`, `f`, `HS` as `constexpr float` and pre-fold into `coeffs`
**Hypothesis**: At `:127-130` we have `result *= 1/sqrt(2)` then `result -= center * f` then `result *= coeffs`. Algebraically `(diag_sum * 1/sqrt(2) + axial_sum - center * f) * coeffs` - the diag scaling could be folded into `coeffs_diag = coeffs * M_SQRT1_2`. Saves 3 multiplies per site per substep (one per substrate).
**Change**: at `:163` compute `coeffs_diag = coeffs * M_SQRT1_2`; rewrite differential to use it. ~10 lines.
**How to measure**: `smsp__inst_executed_pipe_fma` should drop ~5-10 %. Tiny effect on overall time but cheap.
**Risk**: **FP reassociation** - `(a+b+c+d) * k` vs `a*k + b*k + c*k + d*k` differ in rounding. This *is* exactly the kind of reassociation the user OK'd ("at most differing only by FP reassociation tolerances"). Max relative error per cell per substep ~1 ulp; over 24 substeps x 144k steps it could compound. Validate carefully.
**Validity**: Run end-to-end, check `max(|GPU - GPU_baseline|/|GPU_baseline|) < 1e-4` on each substrate field at t = 12 h.

### Experiment 12 - `__launch_bounds__(1024, 1)` on `diffusionKernel`
**Hypothesis**: Without `__launch_bounds__` the compiler defaults to 64-reg/thread sizing. We know occupancy is 1 block/SM (shmem-limited), so declaring `(1024, 1)` tells ptxas it can use up to 255 regs/thread - might enable more aggressive scheduling. Conversely `(1024, 2)` would force tighter regs and is moot since shmem already limits us to 1.
**Change**: annotate `:135`. 1 line.
**How to measure**: `launch__registers_per_thread` may go up; `smsp__warp_cycles_per_issued_instruction` may go down (more ILP). If regs spill - rollback.
**Risk**: Could hurt if compiler over-aggressively schedules. Trivial revert.
**Validity**: SHA256.

### Experiment 13 - Build with `--maxrregcount=64` (gated)
**Hypothesis**: Volta has 65536 regs/SM; at 1024 threads/block 1 block/SM = 64 regs/thread max for full saturation. If the compiler is over 64 regs we lose. Force the budget. Check `cuobjdump` first to see current reg count.
**Change**: per-target compile flag `--maxrregcount=64` on the diffusion TU only. ~5 lines in `src/emt6ro/diffusion/CMakeLists.txt` (need to confirm path).
**How to measure**: `launch__registers_per_thread` - confirm clamp. Watch `smsp__sass_l1tex_t_sectors_pipe_lsu_mem_local` for spill traffic - if it spikes, abort.
**Risk**: Spills to local memory tank performance. Verify with NCU before committing.
**Validity**: SHA256.

### Experiment 14 - Predicate the 3-way border-mask branch
**Hypothesis**: `:154-161` is a 3-way branch (corner / border / interior). Avg active threads/warp 23.67/32 = 25 % lane waste, partly from this branch. Replace with a ternary: `tmp_grid(dr,dc) = is_corner ? zero : (is_border ? external_levels : lattice(r,c).substrates);` - ptxas predicates cleanly.
**Change**: rewrite `:154-161` as a ternary. ~5 lines. The `sites[nsites++]` push must remain conditional - that's fine, it's already lane-divergent only on interior threads.
**How to measure**: `smsp__thread_inst_executed_pred_on_per_inst_executed` should rise toward 1.0 in this section. Active threads/warp should rise toward 32.
**Risk**: The `lattice(r,c).substrates` load is now unconditional - on border threads `lattice(r,c)` may be out-of-bounds (currently guarded by the branch). MUST verify the load is safe (likely yes given the `+2` border on lattice dims).
**Validity**: SHA256 *and* check no OOB via `compute-sanitizer --tool=memcheck`.

### Experiment 15 - Use `__ldg` for the border-mask read
**Hypothesis**: `b_mask(dr-1, dc-1)` (`:156`) is global memory; it's read-only and read-by-every-thread. `__ldg` routes through the read-only texture cache - on Volta this matters because L1 and tex cache are unified but the read-only path bypasses coherency. Marginal.
**Change**: at `:156` use `__ldg(&b_mask.data[(dr-1)*b_mask.dims.width + (dc-1)])`. ~3 lines.
**How to measure**: `l1tex__t_sectors_pipe_lsu_mem_global_op_ld` vs the tex variant; expect a small shift.
**Risk**: None. `b_mask` is read-only in kernel.
**Validity**: SHA256.

---

## Loop structure / occupancy

### Experiment 16 - Block dim 32x32 -> 16x32 (or 32x16) for small ROIs
**Hypothesis**: `kBlockDimX = kBlockDimY = 32` (`:11-12`) gives 1024 threads/block; at full ROI 55x55 with `kSitesPerThread=4` every thread has work. At smaller ROIs many threads have `nsites=0` and just hit barriers. The growth phase has ROIs as small as 5x5 - 25 useful threads out of 1024. Halving block to 512 threads with same shmem could lift occupancy to 2 blocks/SM (current binding is shmem; check). Won't help saturation but helps early steps.
**Change**: `:11-12` add `kBlockDimX_small/Y_small`; dispatch via two distinct `<<<...>>>` launches selected by ROI; or just change globally and measure regression at saturation. ~30 lines for the conditional version, ~5 lines flat.
**How to measure**: `sm__warps_active.avg.pct_of_peak_sustained_active` - expect rise for the first ~30k steps; `gpu__time_duration` per launch breakdown.
**Risk**: At saturation 16x16=256 threads x kSitesPerThread=4 covers only 256x4=1024 sites; full ROI 55x55=3025 - not enough. So conditional dispatch is required, or `kSitesPerThread` must scale.
**Validity**: SHA256 of final lattice.

### Experiment 17 - Block dim 32x32 -> 16x64 / 64x16
**Hypothesis**: Same 1024 threads but different memory access pattern. The current `GRID_FOR` walks columns within a warp (`threadIdx.x` = innermost). 16x64 puts 64 threads in a row, all hitting consecutive shmem columns - better coalesced shmem access if row stride has good bank distribution. Pairs with Experiment 4.
**Change**: change `:11-12` constants and validate `kBlockDiv` (`:13`). ~3 lines.
**How to measure**: bank-conflict counter; L1 SOL.
**Risk**: `kBlockDiv = (55+kBlockDimX-1)/kBlockDimX` - with kBlockDimX=64, kBlockDiv=1 - means each thread owns at most 1 site in X. Need to also adjust kBlockDimY to balance. May reshape the algorithm boundaries.
**Validity**: SHA256.

### Experiment 18 - Sort `sites[]` per thread by (r,c) to maximise shmem locality across substeps
**Hypothesis**: Each thread accumulates up to 4 sites in `sites[]` in load-order. Within a substep, accessing `sites[0]` then `sites[1]` may jump far in shmem. Sort by row-major. ~20-cycle one-shot cost amortised over `steps-1` substeps.
**Change**: post-load, sort 4 elements (3 compare-swaps). ~10 lines around `:162`.
**How to measure**: L1/TEX cycles; expect small improvement.
**Risk**: Trivial. FP-bit-identical (no FP ops affected).
**Validity**: SHA256.

### Experiment 19 - Hoist `sites[i].r/c` out of the inner loop via SoA arrays
**Hypothesis**: `Coords sites[4]` is AoS - `sites[i].r` then `sites[i].c` are two separate `LDS.16` (or worse, two LD.local if spilled). Split into `int16_t sites_r[4], sites_c[4]`. ~3 lines.
**Change**: at `:146` split; update `:160, :167, :169-170`. ~10 lines.
**How to measure**: `smsp__sass_l1tex_t_sectors_pipe_lsu_mem_local` - hopefully zero either way (4-elem arrays should stay in regs). SASS check.
**Risk**: Trivial.
**Validity**: SHA256.

---

## Build flags

### Experiment 20 - Disable `CUDA_SEPARABLE_COMPILATION` for the diffusion TU
**Hypothesis**: `src/CMakeLists.txt:21` and `src/emt6ro/CMakeLists.txt:19` both set `CUDA_SEPARABLE_COMPILATION ON` (and `CMakeLists.txt:10` adds `-dc`). This blocks inlining of `Substrates::operator+=` and `diffusion_differential` across TUs - they're defined inline in headers, so the impact is limited, but `-dc` does change device-link semantics. Try OFF on just the diffusion lib.
**Change**: a `set_target_properties(diffusion PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)` override. ~5 lines.
**How to measure**: SASS diff. Watch `smsp__inst_executed.sum` for any reduction. Modest.
**Risk**: May fail to link if any kernel takes a device-callable function pointer across TUs - test build.
**Validity**: SHA256.

### Experiment 21 - Add `--use_fast_math` gated behind `EMT6RO_FAST_MATH`
**Hypothesis**: `diffusion.cu:34` uses `sqrtf` and `ceilf`; the diffusion kernel itself doesn't call them, but `fillBorderMask` does (`:21`) - called only during `findROIs`, not in the hot path. Within `diffusion_differential` the only operations are `+, -, *` on floats - `--use_fast_math` *may* enable FMA contraction (already happens at `-O3`) and changes denormal handling. Marginal but free.
**Change**: gated flag added to `CMakeLists.txt`. ~6 lines.
**How to measure**: end-to-end wall time. Watch `smsp__sass_inst_executed_op_ffma` rise (FFMA fusion).
**Risk**: Changes FP results. Need to re-validate against C++ ground truth. The user explicitly OK'd reassociation-bound deviation - this qualifies.
**Validity**: End-to-end living-cell counts within 2 % of baseline across 5 protocols x 5 tumors.

### Experiment 22 - Explicit `-O3 -lineinfo` for CUDA Release
**Hypothesis**: Cache shows `CMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG` (cmake/CMakeCache.txt:74). But `CMAKE_BUILD_TYPE` may default to empty for some configs. Make it explicit and add `-lineinfo` for NCU source view (already used in this analysis - confirm it's in the perf build).
**Change**: `add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-O3 -lineinfo>)`. ~3 lines.
**How to measure**: just confirm SASS quality is the same as baseline (we're already on `-O3`); `-lineinfo` is for profiling only.
**Risk**: None.
**Validity**: SHA256 (none expected from `-lineinfo`).

### Experiment 23 - Add `-Xptxas -dlcm=ca` (cache-all loads in L1)
**Hypothesis**: Default for global loads on sm_70 is `cg` (cache in L2 only). The border-mask global read at `:156` and the substrate read at `:159` would benefit from L1 caching across substeps (well, the lattice read is only once at load). Modest; the read-only `__ldg` of Experiment 15 is the cleaner alternative.
**Change**: per-target compile flag. ~3 lines.
**How to measure**: L1 hit rate on global loads.
**Risk**: L1 pressure increases - we're already at 53 % L1 SOL. Could backfire.
**Validity**: SHA256.

---

## sm_80+ only (RTX 5090 forward-looking, gate behind `__CUDA_ARCH__ >= 800`)

### Experiment 24 - `cp.async` for the bordered ROI load into shmem
**Hypothesis**: The load at `:150-162` is a synchronous global -> shmem copy (read into register, write to shmem). On sm_80+, `cuda::memcpy_async` / `__pipeline_memcpy_async` does this with zero register intermediate and can overlap with the first substep's compute. Saves ~one warp's worth of staging.
**Change**: `#if __CUDA_ARCH__ >= 800` block around the load loop. ~30 lines.
**How to measure**: kernel duration; `smsp__warp_issue_stalled_long_scoreboard` should drop.
**Risk**: Need to ensure `lattice(r,c).substrates` is on a 4/8/16-byte boundary - `Site` is 56 B, internal alignment of `substrates` field matters. Verify with `offsetof(Site, substrates)`.
**Validity**: SHA256 on sm_80+ build; on sm_70 path is unchanged.

### Experiment 25 - Use Hopper/Blackwell async barriers (`__pipeline_wait_prior`)
**Hypothesis**: Pairs with 24. The `__syncthreads()` at `:164` becomes a pipeline drain. Only meaningful with 24.
**Change**: included as part of 24's diff.
**How to measure**: see 24.
**Risk / Validity**: see 24.

### Experiment 26 - `__shfl_xor_sync` to share neighbour reads across lanes
**Hypothesis**: Adjacent lanes own adjacent shmem columns; `tmp_grid(r,c-1)` of lane x equals `tmp_grid(r,c)` of lane x-1, so `__shfl_up_sync` could replace one shmem read. But `kSitesPerThread=4` with `kBlockDiv=2` means owned columns are 32 apart - lane-adjacency breaks. Only viable if block dim changes (Experiment 17) re-aligns owned sites to lanes.
**Change**: significant refactor; defer.
**Risk / Validity**: deferred.

---

## Sanity / instrumentation

### Experiment 27 - Verify shmem-per-block via `launch__shared_mem_per_block`
**Hypothesis**: 41 KB is a calculation from `(55+4)^2 * 12 B`. NCU has the direct counter - confirm. If actual is 36 KB or 48 KB, occupancy calculus shifts.
**Change**: rerun NCU with `--section LaunchStats`. Research only.

### Experiment 28 - NCU stall-reason breakdown
**Hypothesis**: We have aggregate "no-eligible 53 %" but not Barrier vs LongSB vs ShortSB. Run with `--section SourceCounters` and `smsp__warp_issue_stalled_*` to discriminate Experiment 7 (barrier) vs Experiment 4 (shmem latency).
**Change**: profiling only.

### Experiment 29 - Annotate NVTX ranges with ROI size
**Hypothesis**: Diffusion cost is ROI^2; need to confirm saturated-ROI tail dominates the 144 k-step run. Codebase already has `EMT6RO_NVTX`; add ROI payload.
**Change**: ~10 lines. Research only.

---

## RECOMMENDED EXPERIMENT ORDER (top 5 by impact/effort)

1. **Experiment 7 - Reduce barrier count from 2 to 1 per substep.** This is the single highest-leverage change available: 53 % no-eligible cycles is almost certainly dominated by `__syncthreads` barrier waits, and the substep loop has 2 of them per iteration. ~15-line diff, likely 10-25 % off diffusion total.
2. **Experiment 4 - Pad shmem row stride to avoid bank conflicts.** L1/TEX SOL 53 % is the highest active throughput - confirm and remove bank conflicts. ~10 lines, NCU `data_bank_conflicts` is a direct read.
3. **Experiment 14 - Predicate the 3-way border-mask branch.** Targets the 25 % lane waste directly. ~5 lines, validates with `compute-sanitizer`.
4. **Experiment 10 + 19 - `#pragma unroll` + sites array hoisting.** Trivial, FP-bit-identical, kills loop control overhead. ~12 lines combined.
5. **Experiment 21 - `--use_fast_math` gated.** Free if calibration tolerates it. Combine with Experiment 11 (pre-fold `M_SQRT1_2` into `coeffs`) for a 5-10 % FMA-pipeline boost; both behind the same FP-tolerance flag.

Experiments 27 + 28 (re-run NCU with finer sections) should run *before* committing to 7 vs 4 - the stall-reason breakdown tells us which of those two has the larger lever.

Surprise candidate: **Experiment 14** (border-mask predication). The user called out divergence as one of several factors, but at 23.67/32 active threads it's worth ~25 % of issue capacity - bigger than the bank-conflict-suspect L1 traffic and almost as big as the barrier-stall hypothesis. It also has the smallest diff of the top-5 picks and is essentially risk-free apart from one OOB-check.
