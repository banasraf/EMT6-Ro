#!/usr/bin/env bash
# Nsight Compute drill-down per hot kernel. One pass per kernel name so
# the report file lets you compare them side-by-side without re-sectioning.
#
# Requires:
#   - ncu on PATH (CUDA toolkit)
#   - the .so built with -DEMT6RO_NVTX=ON (optional) and a Release build
#
# Usage (run from repo root):
#   bash tools/profile_ncu.sh
#   STEPS=2000 BATCH=128 bash tools/profile_ncu.sh
#
# Writes one .ncu-rep per kernel into:
#   runs/perf-baseline-<date>-<gpu>/ncu_<kernel>.ncu-rep
#
# ncu replays each launch multiple times to gather every metric in the
# `full` section set — that means total wall time of this script will be
# several minutes even at small batch. Cap with --launch-count so we
# only profile a handful of launches per kernel.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

STEPS="${STEPS:-5000}"
BATCH="${BATCH:-120}"   # must be a multiple of n_tumors (=10)
SKIP="${SKIP:-20}"      # skip first N launches per kernel (warmup)
COUNT="${COUNT:-3}"     # profile this many launches per kernel
# At 5000 steps: diffuse/simulateCells fire 5000×, updateROIs 156×,
# findOccupied 39×, countLiving 1× — SKIP=20 fits all but countLiving.

# Resolve absolute paths up front. sudo's secure_path strips
# /usr/local/cuda/bin so root won't find ncu / python3 otherwise.
NCU="$(command -v ncu)"
PY3="$(command -v python3)"
if [[ -z "$NCU" ]]; then
  echo "ncu not found on PATH; is the CUDA toolkit installed?" >&2
  exit 1
fi

DATE="$(date +%Y-%m-%d)"
GPU="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 \
  | tr '[:upper:]' '[:lower:]' \
  | sed -e 's/nvidia//g' -e 's/geforce//g' -e 's/[^a-z0-9]//g')"
GPU="${GPU:-gpu}"
OUTDIR="runs/perf-baseline-${DATE}-${GPU}"
mkdir -p "$OUTDIR"

# Kernel regex patterns. Names come from the CUDA source — verify with
# `cuobjdump --dump-elf-symbols path/to/.so | grep -E 'Kernel|simulation'`.
# countLivingKernel is excluded: it fires once per getResults() and is
# 0.0 % of the baseline; not worth a profile pass.
KERNELS=(
  "cellSimulationKernel"
  "diffusionKernel"
  "findROIsKernel"
  "findOccupied"          # not suffixed with Kernel in source
)

for K in "${KERNELS[@]}"; do
  OUT="${OUTDIR}/ncu_${K}.ncu-rep"
  echo "ncu -> ${OUT}  (skip=${SKIP} count=${COUNT})"
  # sudo -E preserves PYTHONPATH; ncu requires CAP_SYS_ADMIN equivalent
  # (NVreg_RestrictProfilingToAdminUsers=1 by default on this driver).
  # Absolute paths because sudo's secure_path doesn't include /usr/local/cuda/bin.
  sudo -E env PYTHONPATH="$REPO/python" "$NCU" \
    --set full \
    --kernel-name "regex:${K}" \
    --launch-skip "$SKIP" \
    --launch-count "$COUNT" \
    --force-overwrite \
    --export "$OUT" \
    "$PY3" tools/perf_baseline.py --steps "$STEPS" --batch-size "$BATCH" --repeats 1 --warmup-steps 0 --no-write \
    || echo "  (kernel ${K}: ncu returned non-zero — check above)"
done

echo
echo "Reports in: ${OUTDIR}/ncu_*.ncu-rep"
echo "Open with:  ncu-ui ${OUTDIR}/ncu_cellSimulationKernel.ncu-rep"
