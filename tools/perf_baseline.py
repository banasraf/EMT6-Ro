#!/usr/bin/env python3
"""
Reproducible performance baseline for the EMT6-Ro CUDA engine.

Runs one canonical 10-day workload (144 000 steps, fixed seed and protocol)
at the largest batch_size that fits in device memory and emits a Markdown
report + machine-readable JSON under
runs/perf-baseline-YYYY-MM-DD-<gpu-slug>/.

Captures: git HEAD, GPU model, driver, CUDA runtime, CPU model, batch_size,
wall-time per repeat (min/median/max), sims/sec, and the per-kernel
breakdown via Experiment.get_kernel_timers() — which is non-zero only in
builds with -DEMT6RO_TIMING=ON.

Usage (run from the repo root after building the .so into python/):

    PYTHONPATH=$PWD/python python3 tools/perf_baseline.py
    PYTHONPATH=$PWD/python python3 tools/perf_baseline.py --batch-size 256
    PYTHONPATH=$PWD/python python3 tools/perf_baseline.py --quick   # 5000 steps
"""

import argparse
import json
import os
import platform
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).parent.resolve()
REPO = HERE.parent
sys.path.insert(0, str(REPO / "python"))

import numpy as np
from emt6ro.simulation import Experiment, load_parameters, load_state

# Canonical irradiation protocol — lifted from benchmarks.py:15-18
# (bm1: 1.25 Gy at hours 0,6,24,30,48,54,72,78).
CANONICAL_PROTOCOL = [
    (0,         1.25), (6 * 600,    1.25),
    (24 * 600,  1.25), (30 * 600,   1.25),
    (48 * 600,  1.25), (54 * 600,   1.25),
    (72 * 600,  1.25), (78 * 600,   1.25),
]


def gpu_info():
    """Return dict with GPU name, driver, total/free MB. Empty on failure."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.free",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        # Take first GPU only.
        name, driver, total_mb, free_mb = [s.strip() for s in out.splitlines()[0].split(",")]
        return {
            "name": name,
            "driver": driver,
            "memory_total_mb": int(total_mb),
            "memory_free_mb": int(free_mb),
        }
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {}


def cpu_info():
    try:
        out = Path("/proc/cpuinfo").read_text()
        m = re.search(r"^model name\s*:\s*(.+)$", out, re.MULTILINE)
        if m:
            return m.group(1).strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def git_head():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unknown"


def cuda_runtime():
    try:
        out = subprocess.check_output(
            ["nvcc", "--version"], stderr=subprocess.DEVNULL).decode()
        m = re.search(r"release\s+(\d+\.\d+)", out)
        return m.group(1) if m else "unknown"
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unknown"


def gpu_slug(name):
    # "NVIDIA GeForce RTX 5090" -> "rtx5090"; "NVIDIA TITAN V" -> "titanv"
    s = name.lower()
    s = re.sub(r"\bnvidia\b", "", s)
    s = re.sub(r"\bgeforce\b", "", s)
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s or "gpu"


# Per-sim memory cost (approximate).
# Constituents in Simulation::Simulation (src/emt6ro/simulation/simulation.cu):
#   data            53*53*sizeof(Site) ~ 168 KB
#   border_masks    53*53*1            ~ 2.8 KB
#   occupied        1024*4              = 4 KB
#   rand_state      512*48              ~ 24 KB
#   protocol slots, lattice views, rois, results: negligible
# ~ 200 KB/sim. Headroom factor 1.4 to cover Experiment overhead +
# pybind buffers + working set. Round to 280 KB/sim.
PER_SIM_KB = 280


def autosize_batch(n_tumors, free_mb, headroom_frac=0.70):
    """Largest multiple of n_tumors that fits in (free_mb * headroom_frac)."""
    budget_kb = free_mb * 1024 * headroom_frac
    n = int(budget_kb // PER_SIM_KB)
    n -= n % n_tumors  # divisible by n_tumors so runs/tumor is integer
    return max(n_tumors, n)


def run_one(params, tumors, protocol, batch_size, n_steps, seed):
    """Construct an Experiment, run nsteps, return (wall_sec, kernel_timers, final_counts)."""
    runs = batch_size // len(tumors)
    exp = Experiment(
        params, tumors,
        runs=runs, protocols_num=1,
        simulation_steps=n_steps, protocol_resolution=300,
    )
    exp.add_irradiations([protocol])
    exp._experiment.reset_kernel_timers()
    t0 = time.time()
    exp.run(n_steps)
    results = np.asarray(exp.get_results())
    t1 = time.time()
    timers = exp._experiment.kernel_timers()
    return t1 - t0, timers, results


def timers_dict(t):
    return {
        "findOccupied_ms": float(t.findOccupied_ms),
        "updateROIs_ms": float(t.updateROIs_ms),
        "diffuse_ms": float(t.diffuse_ms),
        "simulateCells_ms": float(t.simulateCells_ms),
        "countLiving_ms": float(t.countLiving_ms),
        "n_steps": int(t.n_steps),
    }


def write_report(out_dir, payload):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run.json").write_text(json.dumps(payload, indent=2))

    p = payload
    t_med = p["repeats"]["median_wall_sec"]
    sps_med = p["repeats"]["median_sims_per_sec"]
    tk = p["kernel_timers_repeat_0"]
    total_ms = sum(tk[k] for k in ("findOccupied_ms", "updateROIs_ms",
                                   "diffuse_ms", "simulateCells_ms",
                                   "countLiving_ms"))

    def pct(x):
        return f"{(x / total_ms * 100.0):.1f}%" if total_ms > 0 else "n/a"

    md = []
    md.append(f"# EMT6-Ro perf baseline — {p['timestamp']}")
    md.append("")
    md.append("## Environment")
    md.append(f"- GPU: **{p['gpu']['name']}** ({p['gpu'].get('memory_total_mb','?')} MB total, "
              f"{p['gpu'].get('memory_free_mb','?')} MB free at start)")
    md.append(f"- Driver: {p['gpu'].get('driver','?')} · CUDA runtime: {p['cuda_runtime']}")
    md.append(f"- CPU: {p['cpu']}")
    md.append(f"- Git: `{p['git']}`")
    md.append(f"- Built with EMT6RO_TIMING: **{'yes' if total_ms > 0 else 'no (kernel timers all zero)'}**")
    md.append("")
    md.append("## Workload")
    md.append(f"- Steps per sim: **{p['workload']['steps']}** "
              f"({'10 days' if p['workload']['steps'] == 144000 else 'custom'})")
    md.append(f"- batch_size: **{p['workload']['batch_size']}** "
              f"({p['workload']['n_tumors']} tumors × "
              f"{p['workload']['runs_per_tumor']} reps × 1 protocol)")
    md.append(f"- Protocol: {p['workload']['protocol']}")
    md.append(f"- Warmup steps: {p['workload']['warmup_steps']} (discarded)")
    md.append(f"- Timed repeats: {p['workload']['repeats']}")
    md.append("")
    md.append("## Result")
    md.append("")
    md.append("| repeat | wall (s) | sims/sec |")
    md.append("|--------|----------|----------|")
    for i, (w, s) in enumerate(zip(p["repeats"]["wall_sec"], p["repeats"]["sims_per_sec"])):
        md.append(f"| {i} | {w:.3f} | {s:.1f} |")
    md.append("")
    md.append(f"- **median wall**: {t_med:.3f} s")
    md.append(f"- **median sims/sec**: {sps_med:.1f}")
    md.append(f"- **sims/sec/GB (median)**: "
              f"{sps_med / max(1, p['gpu'].get('memory_total_mb', 1) / 1024):.2f}")
    md.append("")
    md.append("## Per-kernel breakdown (repeat 0)")
    md.append("")
    if total_ms > 0:
        md.append("| kernel | ms | share |")
        md.append("|--------|----|-------|")
        md.append(f"| diffuse | {tk['diffuse_ms']:.1f} | {pct(tk['diffuse_ms'])} |")
        md.append(f"| simulateCells | {tk['simulateCells_ms']:.1f} | {pct(tk['simulateCells_ms'])} |")
        md.append(f"| updateROIs | {tk['updateROIs_ms']:.1f} | {pct(tk['updateROIs_ms'])} |")
        md.append(f"| findOccupied | {tk['findOccupied_ms']:.1f} | {pct(tk['findOccupied_ms'])} |")
        md.append(f"| countLiving | {tk['countLiving_ms']:.1f} | {pct(tk['countLiving_ms'])} |")
        md.append(f"| **total** | **{total_ms:.1f}** | 100% |")
        md.append("")
        md.append("Sum of kernel ms includes per-call host sync overhead "
                  "(EMT6RO_TIMING uses cudaEventSynchronize after each launch); "
                  "this is the price of the per-kernel breakdown. Compare with "
                  "the wall column — gaps are host overhead + non-kernel work.")
    else:
        md.append("_Per-kernel times are zero — rebuild with "
                  "`-DEMT6RO_TIMING=ON` for the breakdown._")
    md.append("")
    md.append("## Reproduce")
    md.append("```bash")
    md.append(f"git checkout {p['git']}")
    md.append("mkdir -p build && cd build")
    md.append("CC=gcc-11 CXX=g++-11 cmake -DCMAKE_BUILD_TYPE=Release "
              "-DCMAKE_CUDA_ARCHITECTURES=70 -DEMT6RO_TIMING=ON -DEMT6RO_NVTX=ON ..")
    md.append("make -j$(nproc)")
    md.append("cd ..")
    md.append("PYTHONPATH=$PWD/python python3 tools/perf_baseline.py "
              f"--batch-size {p['workload']['batch_size']}")
    md.append("```")
    (out_dir / "REPORT.md").write_text("\n".join(md) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--params", type=Path, default=REPO / "data/default-parameters.json")
    ap.add_argument("--tumor-dir", type=Path, default=REPO / "data/tumor-lib")
    ap.add_argument("--steps", type=int, default=144000, help="simulation length")
    ap.add_argument("--quick", action="store_true",
                    help="Override --steps with 5000 (fast smoke / nsys capture)")
    ap.add_argument("--warmup-steps", type=int, default=1000)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=None,
                    help="Override autosized batch (must be a multiple of 10)")
    ap.add_argument("--out-root", type=Path, default=REPO / "runs")
    ap.add_argument("--no-write", action="store_true",
                    help="Skip writing REPORT.md / run.json (used when "
                         "perf_baseline is invoked under nsys / ncu so the "
                         "real saturation baseline isn't overwritten).")
    args = ap.parse_args(argv)

    if args.quick:
        args.steps = 5000

    gi = gpu_info()
    if not gi:
        print("WARNING: nvidia-smi not available — cannot autosize batch.",
              file=sys.stderr)

    print(f"Loading params: {args.params}")
    params = load_parameters(str(args.params))
    tumor_files = sorted(args.tumor_dir.glob("tumor-*.txt"))
    if not tumor_files:
        sys.exit(f"No tumor-*.txt files in {args.tumor_dir}")
    tumors = [load_state(str(p), params) for p in tumor_files]
    n_tumors = len(tumors)
    print(f"Loaded {n_tumors} tumors from {args.tumor_dir}")

    if args.batch_size is None:
        if not gi:
            sys.exit("Cannot autosize batch (no nvidia-smi). Pass --batch-size.")
        batch_size = autosize_batch(n_tumors, gi["memory_free_mb"])
    else:
        if args.batch_size % n_tumors:
            sys.exit(f"--batch-size must be a multiple of n_tumors={n_tumors}")
        batch_size = args.batch_size
    runs_per_tumor = batch_size // n_tumors
    print(f"batch_size = {batch_size}  ({n_tumors} tumors × {runs_per_tumor} reps)")

    # Warmup: short run, results discarded. Exercises the JIT + caches.
    if args.warmup_steps > 0:
        print(f"Warmup: {args.warmup_steps} steps ...")
        w0 = time.time()
        run_one(params, tumors, CANONICAL_PROTOCOL, batch_size,
                args.warmup_steps, seed=0)
        print(f"  done in {time.time()-w0:.2f}s")

    wall_secs = []
    sims_per_sec = []
    first_timers = None
    for i in range(args.repeats):
        print(f"Repeat {i}: {args.steps} steps × {batch_size} sims ...")
        wall, timers, _ = run_one(
            params, tumors, CANONICAL_PROTOCOL, batch_size, args.steps, seed=i)
        sps = batch_size / wall
        wall_secs.append(wall)
        sims_per_sec.append(sps)
        if first_timers is None:
            first_timers = timers
        print(f"  wall={wall:.2f}s  sims/sec={sps:.1f}")

    gpu_name = gi.get("name", "unknown")
    out_dir = args.out_root / f"perf-baseline-{datetime.now().strftime('%Y-%m-%d')}-{gpu_slug(gpu_name)}"
    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "git": git_head(),
        "gpu": gi,
        "cpu": cpu_info(),
        "cuda_runtime": cuda_runtime(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "workload": {
            "steps": args.steps,
            "batch_size": batch_size,
            "n_tumors": n_tumors,
            "runs_per_tumor": runs_per_tumor,
            "warmup_steps": args.warmup_steps,
            "repeats": args.repeats,
            "protocol": [[int(t), float(d)] for t, d in CANONICAL_PROTOCOL],
        },
        "repeats": {
            "wall_sec": wall_secs,
            "sims_per_sec": sims_per_sec,
            "median_wall_sec": float(np.median(wall_secs)),
            "median_sims_per_sec": float(np.median(sims_per_sec)),
            "min_wall_sec": float(np.min(wall_secs)),
            "max_wall_sec": float(np.max(wall_secs)),
        },
        "kernel_timers_repeat_0": timers_dict(first_timers),
    }
    if args.no_write:
        print("\n--no-write: skipping REPORT.md / run.json")
    else:
        write_report(out_dir, payload)
        print(f"\nWrote {out_dir}/REPORT.md")
        print(f"      {out_dir}/run.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
