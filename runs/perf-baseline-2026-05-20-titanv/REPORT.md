# EMT6-Ro perf baseline — 2026-05-20T15:38:38

## Environment
- GPU: **NVIDIA TITAN V** (12288 MB total, 12012 MB free at start)
- Driver: 555.42.06 · CUDA runtime: 12.5
- CPU: Intel(R) Core(TM) i7-7800X CPU @ 3.50GHz
- Git: `1d9736d7a50c4c6182690bf6d630a6b5d609296e`
- Built with EMT6RO_TIMING: **yes**

## Workload
- Steps per sim: **144000** (10 days)
- batch_size: **30750** (10 tumors × 3075 reps × 1 protocol)
- Protocol: [[0, 1.25], [3600, 1.25], [14400, 1.25], [18000, 1.25], [28800, 1.25], [32400, 1.25], [43200, 1.25], [46800, 1.25]]
- Warmup steps: 1000 (discarded)
- Timed repeats: 1

## Result

| repeat | wall (s) | sims/sec |
|--------|----------|----------|
| 0 | 4459.647 | 6.9 |

- **median wall**: 4459.647 s
- **median sims/sec**: 6.9
- **sims/sec/GB (median)**: 0.57

## Per-kernel breakdown (repeat 0)

| kernel | ms | share |
|--------|----|-------|
| diffuse | 3037126.8 | 68.1% |
| simulateCells | 1366625.0 | 30.7% |
| updateROIs | 43686.0 | 1.0% |
| findOccupied | 10007.9 | 0.2% |
| countLiving | 7.5 | 0.0% |
| **total** | **4457453.2** | 100% |

Sum of kernel ms includes per-call host sync overhead (EMT6RO_TIMING uses cudaEventSynchronize after each launch); this is the price of the per-kernel breakdown. Compare with the wall column — gaps are host overhead + non-kernel work.

## Reproduce
```bash
git checkout 1d9736d7a50c4c6182690bf6d630a6b5d609296e
mkdir -p build && cd build
CC=gcc-11 CXX=g++-11 cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=70 -DEMT6RO_TIMING=ON -DEMT6RO_NVTX=ON ..
make -j$(nproc)
cd ..
PYTHONPATH=$PWD/python python3 tools/perf_baseline.py --batch-size 30750
```
