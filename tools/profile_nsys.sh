#!/usr/bin/env bash
# Capture a Nsight Systems timeline of the EMT6-Ro perf_baseline harness.
#
# Requires:
#   - nsys on PATH (CUDA toolkit or stand-alone install)
#   - the .so built with -DEMT6RO_NVTX=ON (and ideally -DEMT6RO_TIMING=ON)
#
# Usage (run from repo root):
#   bash tools/profile_nsys.sh                 # quick (5 000 steps)
#   bash tools/profile_nsys.sh --steps 20000   # custom length
#   STEPS=20000 BATCH=512 bash tools/profile_nsys.sh
#
# Writes the .nsys-rep (and .qdrep alias on older nsys) into
#   runs/perf-baseline-<date>-<gpu>/nsys.{nsys-rep,sqlite}
# Open in Nsight Systems GUI: `nsys-ui runs/.../nsys.nsys-rep`

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

STEPS="${STEPS:-5000}"
BATCH="${BATCH:-}"
EXTRA_ARGS=("$@")

DATE="$(date +%Y-%m-%d)"
GPU="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 \
  | tr '[:upper:]' '[:lower:]' \
  | sed -e 's/nvidia//g' -e 's/geforce//g' -e 's/[^a-z0-9]//g')"
GPU="${GPU:-gpu}"
OUTDIR="runs/perf-baseline-${DATE}-${GPU}"
mkdir -p "$OUTDIR"

PY_ARGS=(--quick --no-write)
if [[ "$STEPS" != "5000" ]]; then
  PY_ARGS=(--steps "$STEPS" --no-write)
fi
if [[ -n "$BATCH" ]]; then
  PY_ARGS+=(--batch-size "$BATCH")
fi
PY_ARGS+=("${EXTRA_ARGS[@]}")

echo "nsys profile -> ${OUTDIR}/nsys"
echo "  steps=${STEPS}  batch=${BATCH:-auto}"

# -t cuda,nvtx,osrt: capture CUDA API, NVTX ranges, and OS runtime events.
# --capture-range=cudaProfilerApi works if the harness ever calls
# cudaProfilerStart; here we just capture the whole process.
PYTHONPATH="$REPO/python" nsys profile \
  --trace=cuda,nvtx,osrt \
  --cuda-memory-usage=true \
  --output="${OUTDIR}/nsys" \
  --force-overwrite=true \
  python3 tools/perf_baseline.py "${PY_ARGS[@]}"

echo
echo "Report:    ${OUTDIR}/nsys.nsys-rep"
echo "Open with: nsys-ui ${OUTDIR}/nsys.nsys-rep"
