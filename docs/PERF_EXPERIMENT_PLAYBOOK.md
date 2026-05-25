# Perf experiment playbook — worktree fan-out + funnel benchmarking

How to run the 29 diffusion-kernel tuning experiments (or any future
backlog of independent semantics-preserving optimisations) on a single
GPU machine without burning days of wall time.

Backlog source: `runs/perf-baseline-2026-05-20-titanv/DIFFUSION_TUNING_EXPERIMENTS.md`

## Why the funnel

A naive plan ("for each experiment, run the full 10-day benchmark with
3 repeats") costs **~4.5 days of GPU time** on Titan V (probably ~1.5-2
days on RTX 5090). Most of that is wasted because:

- You only need long simulations to validate **semantics** (does the
  optimisation change downstream cell counts?). Not to **rank perf**.
- A 500-step run produces 500 diffuse-kernel launches × ~21 ms each =
  ~10 s of stable per-launch signal. Plenty for ranking; per-launch
  noise is well under 1 %.

Two-stage funnel:

| stage | purpose | per-experiment wall | total for 29 |
|-------|---------|---------------------|--------------|
| A — ranking | identify candidates with real signal | ~30-60 s | ~15-30 min |
| B — validation + headline | semantics check + full-length perf claim on top 3-5 | ~10 min (B2) + ~3.7 hr (B3) | ~5 hr |
| **Total** | | | **~5 hr instead of 4.5 days** |

## Layout — one worktree per experiment

```
emt6ro/                    main checkout
emt6ro-exp-01/             git worktree, branch exp/diff-tuning-01
emt6ro-exp-02/             git worktree, branch exp/diff-tuning-02
…
emt6ro-exp-29/
exp-manifest.json          shared state (both phases write to it)
exp-report/
├── quick/                 Stage A artefacts
│   ├── exp-01/run.json
│   └── exp-02/run.json
├── full/                  Stage B artefacts (winners only)
│   └── exp-07/run.json
└── COMPARISON.md          aggregated leaderboard
```

Each worktree gets its own `build/` so per-experiment artefacts (the
`.so`) never collide.

## Phase 1 — Setup (one-time, ~5 min)

```bash
# tools/exp_setup.sh
set -euo pipefail
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT/.."
for i in $(seq -w 1 29); do
  if [[ ! -d "emt6ro-exp-${i}" ]]; then
    (cd "$REPO_ROOT" && git worktree add -b "exp/diff-tuning-${i}" "../emt6ro-exp-${i}" master)
  fi
done

# Emit manifest skeleton — one row per experiment.
cat > "$REPO_ROOT/exp-manifest.json" <<'EOF'
[
  {"id":"ref","branch":"master","worktree":"../emt6ro","build_status":"pending","smoke_status":"pending","quick_status":"pending","full_status":"pending"}
]
EOF
# (Populate with all 29 entries via a small Python script — see exp_setup.py for a fuller version.)
```

## Phase 2 — Implementation (parallel, capped at 4-8 agents)

One subagent per experiment. Prompt template:

```text
You are implementing diffusion-kernel experiment #{N}.

Worktree: /abs/path/to/emt6ro-exp-{N}
Spec (verbatim from DIFFUSION_TUNING_EXPERIMENTS.md):
<… paste the section for experiment #{N} …>

Tasks:
1. Implement the change. Only touch the files cited in the spec.
2. Build:
   mkdir -p build && cd build
   CC=gcc-11 CXX=g++-11 cmake -DCMAKE_BUILD_TYPE=Release \
     -DCMAKE_CUDA_ARCHITECTURES=120 -DEMT6RO_TIMING=ON ..
   make -j$(nproc)
3. Verify backend.*.so was produced. Symlink it into python/emt6ro/simulation/.
4. Smoke test (~5 s):
   PYTHONPATH=$PWD/python python3 -c '
   from emt6ro.simulation import Experiment, load_parameters, load_state
   p = load_parameters("data/default-parameters.json")
   ts = [load_state(f"data/tumor-lib/tumor-{i}.txt", p) for i in range(1, 11)]
   e = Experiment(p, ts, runs=4, protocols_num=1,
                  simulation_steps=2000, protocol_resolution=300)
   e.add_irradiations([[(0, 2.0)]])
   e.run(2000)
   res = e.get_results()
   import numpy as np
   assert np.all(np.isfinite(res)) and res.sum() > 0, "smoke failed"
   print("smoke ok")
   '
5. Update exp-manifest.json (use flock to serialise writes — see below).
6. DO NOT run perf_baseline.py — that's phase 3.

Output your row to exp-manifest.json:
{
  "id": "{N}",
  "branch": "exp/diff-tuning-{N}",
  "worktree": "/abs/path/to/emt6ro-exp-{N}",
  "build_status": "ok" | "failed",
  "build_log_path": "...",
  "so_path": "...",
  "smoke_status": "ok" | "nan" | "crash",
  "loc_changed": <int>,
  "implementation_notes": "<one-line>"
}
```

### Manifest concurrency

```bash
# helper inside each subagent's "update manifest" step
flock /tmp/exp-manifest.lock python3 -c '
import json, sys
m = json.load(open("exp-manifest.json"))
# upsert this experiments row by "id"
row = json.loads(sys.argv[1])
m = [r for r in m if r["id"] != row["id"]] + [row]
m.sort(key=lambda r: r["id"])
json.dump(m, open("exp-manifest.json","w"), indent=2)
' '<json-row>'
```

### Concurrency limit

nvcc peaks around 2-4 GB RAM per parallel translation unit and one
build alone uses ~6 parallel TUs. Cap parallel subagents at:

- 4-6 on a 32 GB box
- 8-10 on a 64 GB+ box

OpenCode's parallel-Task mechanism handles the fan-out; set
`max_concurrent` appropriately.

## Phase 3 Stage A — Quick ranking (sequential, ~30 sec each)

Driver iterates the manifest, runs the harness with short workload:

```bash
# tools/exp_bench_quick.sh
set -euo pipefail
REPO="$(git rev-parse --show-toplevel)"

# Reference run first
echo "Running reference (master HEAD)..."
cd "$REPO"
PYTHONPATH="$PWD/python" python3 tools/perf_baseline.py \
  --steps 500 --warmup-steps 100 --batch-size 30750 --repeats 3 \
  --out-root "$REPO/exp-report/quick/exp-ref"
cd -

# Then each experiment
for row in $(jq -c '.[] | select(.id!="ref" and .build_status=="ok" and .smoke_status=="ok")' "$REPO/exp-manifest.json"); do
  ID=$(jq -r .id <<<"$row")
  WT=$(jq -r .worktree <<<"$row")
  SO=$(jq -r .so_path <<<"$row")
  echo "Running exp-${ID}..."
  ln -sf "$SO" "$WT/python/emt6ro/simulation/backend.cpython-310-x86_64-linux-gnu.so"
  cd "$WT"
  PYTHONPATH="$PWD/python" python3 tools/perf_baseline.py \
    --steps 500 --warmup-steps 100 --batch-size 30750 --repeats 3 \
    --out-root "$REPO/exp-report/quick/exp-${ID}"
  cd -
done
```

**Tune `--batch-size` to the new GPU.** On RTX 5090 (32 GB) the
autosizer will pick something larger than 30 750 — let it autosize once
on the reference run and then hard-code the batch for consistency
across all 29 experiments.

### Important: serial only

Concurrent GPU jobs invalidate kernel timings. The driver is strictly
sequential. Each experiment takes ~30-60 s, so 29 experiments is
~15-30 min total.

## Phase 3 Stage B — Validation + headline (only top 3-5)

Take the leaderboard from Stage A. For each experiment with > 5 %
improvement in `diffuse_ms` and no regression in other kernels:

| step | what | wall | applied to |
|------|------|------|------------|
| B1 | Layer-1 stability (1 sim, 144k steps, check substrates finite & physical) | ~30 s | every Stage-A winner |
| B2 | Distributional KS test (3 × 5 × 30 reps, 144k steps, vs reference) | ~10 min | top 5 |
| B3 | Headline benchmark (full saturating, `--repeats 3`, both TIMING=ON and TIMING=OFF builds) | ~3.7 hr × 2 = ~7.5 hr | top 1-2 |

Scripts: `tools/exp_validate.sh` and `tools/exp_bench_full.sh` (write
when needed; mirror the Stage A driver structure).

### Validation acceptance criteria

| criterion | threshold | meaning |
|-----------|-----------|---------|
| stability | substrates all finite, non-negative, < 10 × `external_levels` max | catches CFL violations |
| KS p-value | > 0.001 across all 15 (tumor, protocol) pairs | distributions plausibly identical |
| Cohen's d | < 0.2 across all 15 pairs | "small effect" — safe for scientific conclusions |
| no other-kernel regression | each non-diffuse kernel within 2 % of reference | optimisation is targeted, not just shifting cost |

## Phase 4 — Report

```bash
# tools/exp_collect.py — aggregates run.json files into a single table
python3 tools/exp_collect.py exp-report/ > exp-report/COMPARISON.md
```

`COMPARISON.md` schema:

| id | name | quick wall (s) | Δ sims/sec | quick diffuse ms | Δ diffuse | other-kernel regress? | KS p | Cohen's d | recommend |
|----|------|----------------|------------|------------------|-----------|----------------------|------|-----------|-----------|
| ref | master | 33.2 | — | 9 800 | — | — | — | — | — |
| 14 | border-mask ternary | 30.8 | +7.3 % | 8 880 | −9.4 % | none | 0.42 | 0.05 | **ship** |
| 07 | re-roll substep | 29.0 | +13.2 % | 8 070 | −17.6 % | +2 % cells | 0.001 ⚠ | 0.31 | **investigate** |
| 04 | shmem padding | 31.6 | +4.6 % | 9 230 | −5.9 % | none | 0.71 | 0.03 | **ship** |

Colour-coded (green/yellow/red) in the rendered output; rendered with
github-flavoured markdown.

## Caveats

- `EMT6RO_TIMING=ON` adds host-sync overhead. **All Stage A experiments
  share it** so deltas are comparable. **Headline numbers (Stage B3)
  must be re-measured with `EMT6RO_TIMING=OFF`** for absolute claims.
- Reference baseline must be on the **same machine** as the experiments.
  Do NOT compare RTX 5090 experiments to the Titan V baseline in
  `runs/perf-baseline-2026-05-20-titanv/`.
- The 29 experiments are individually independent but combining them
  could over- or under-shoot expected gains. After picking ≤5 winners,
  apply them cumulatively (one branch with all winners) and re-bench
  vs reference — that's the final "shipped optimisation" number.

## What's intentionally NOT in this playbook

Algorithmic changes (9-pt → 5-pt stencil, halving substep count,
per-substrate kernel split, SoA Site layout) have their own validation
paths because they change numerical semantics. See `docs/PERF_HANDOFF.md`
for the broader optimisation strategy that includes those.
