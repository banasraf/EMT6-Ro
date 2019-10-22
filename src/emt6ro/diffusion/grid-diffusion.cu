#include "emt6ro/diffusion/grid-diffusion.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_device_runtime_api.h>
#include <cmath>
#include <iostream>
#include <cassert>
#include <algorithm>
#include "emt6ro/common/debug.h"

namespace emt6ro {

__device__ void copyLattice(GridView<Site> &lattice, const ROI &roi,
                            GridView<Substrates> &temp_lattice, const Substrates &ext_levels) {
  const auto h = temp_lattice.dims.height;
  const auto w = temp_lattice.dims.width;
  GRID_FOR(0, 0, h, w) {
    if (r == 0 || r == h - 1 || c == 0 || c == w - 1) {
      temp_lattice(r, c) = ext_levels;
    } else {
      temp_lattice(r, c) = lattice(roi.origin.r + r - 1, roi.origin.c + c  - 1).substrates;
    }
  }
}

__device__ void copyLatticeBack(GridView<Site> &lattice, const ROI &roi,
                                GridView<Substrates> temp_lattice) {
  GRID_FOR(1, 1, temp_lattice.dims.height - 1, temp_lattice.dims.width - 1) {
    lattice(roi.origin.r + r - 1, roi.origin.c + c - 1).substrates = temp_lattice(r, c);
  }
}

__host__ __device__ Substrates paramDiffusion(Substrates levels, Substrates coeffs, float time_step,
                                              Substrates ortho_sum, Substrates diag_sum) {
  constexpr float HS = 2 * M_SQRT2f32;
  constexpr float f = 4. + HS;
  return (coeffs*time_step*HS)/f * (ortho_sum + diag_sum*M_SQRT1_2f32 - levels * f) + levels;
}

__device__ void diffuse(GridView<Substrates> &out, const GridView<Substrates> &in,
                        const Substrates &coeffs, float time_step) {
  const auto h  = out.dims.height;
  const auto w = out.dims.width;
  GRID_FOR(1, 1, h-1, w-1) {
      auto diag = in(r - 1, c - 1) +
                  in(r - 1, c + 1) +
                  in(r + 1, c - 1) +
                  in(r + 1, c + 1);
      auto ortho = in(r - 1, c) +
                   in(r, c - 1) +
                   in(r + 1, c) +
                   in(r, c + 1);
      out(r, c) = paramDiffusion(in(r, c), coeffs, time_step, ortho, diag);
  }
}

__device__ void diffusion(GridView<Substrates> &lattice1, GridView<Substrates> &lattice2,
                          const Substrates &coeffs, float time_step, uint32_t steps) {
  for (uint32_t s = 0; s < steps / 2; ++s) {
    diffuse(lattice2, lattice1, coeffs, time_step);
    __syncthreads();
    diffuse(lattice1, lattice2, coeffs, time_step);
    __syncthreads();
  }
}

__global__ void diffKernel(GridView<Site> *lattices, const ROI *rois, Substrates *temp_mem,
                           Substrates coeffs, float time_step, uint32_t steps,
                           Substrates ext_levels) {
  extern __shared__ Substrates shared_mem[];
  const auto bid = blockIdx.x;
  const auto t_h = rois[bid].dims.height + 2;
  const auto t_w = rois[bid].dims.width + 2;
  GridView<Substrates> temp_lattice{temp_mem + t_h*t_w*bid, Dims{t_h, t_w}};
  GridView<Substrates> shared_lattice{shared_mem, Dims{t_h, t_w}};
  GRID_FOR(0, 0, t_h, t_w) {
    shared_lattice(r, c) = ext_levels;
  }
  copyLattice(lattices[bid], rois[bid], temp_lattice, ext_levels);
  __syncthreads();
  diffusion(temp_lattice, shared_lattice, coeffs, time_step, steps);
  copyLatticeBack(lattices[bid], rois[bid], temp_lattice);
}

void batchDiffuse(GridView<Site> *lattices, const ROI *rois, Substrates *temp_mem,
                   Dims max_dims, const Substrates &coeffs, const Substrates &ext_levels,
                   uint32_t batch_size, float time_step, uint32_t steps) {
  assert(steps % 2 == 0);
  const auto b_dims = Dims{max_dims.height + 2, max_dims.width + 2};
  auto box_h = b_dims.height < CuBlockDimY ? b_dims.height : CuBlockDimY;
  auto box_w = b_dims.width < CuBlockDimX ? b_dims.width : CuBlockDimX;
  diffKernel<<<batch_size, dim3(box_w, box_h), b_dims.vol() * sizeof(Substrates)>>>
    (lattices, rois, temp_mem, coeffs, time_step, steps, ext_levels);
  KERNEL_DEBUG("diffuse")
}

struct MinMax {
  int32_t min;
  int32_t max;
};

__device__ void reduce2d_min(GridView<MinMax> &grid, int32_t r, int32_t c, int32_t d) {
  grid(r, c).min = min(grid(r, c).min, grid(r, c + d).min);
  grid(r + d, c).min = min(grid(r + d, c).min, grid(r + d, c + d).min);
  grid(r, c).min = min(grid(r, c).min, grid(r + d, c).min);
}

__device__ void reduce2d_max(GridView<MinMax> &grid, int32_t r, int32_t c, int32_t d) {
  grid(r, c).max = max(grid(r, c).max, grid(r, c + d).max);
  grid(r + d, c).max = max(grid(r + d, c).max, grid(r + d, c + d).max);
  grid(r, c).max = max(grid(r, c).max, grid(r + d, c).max);
}

__global__ void findBoundariesKernel(const GridView<Site> *lattices, ROI *rois) {
  int32_t bh = blockDim.y;
  int32_t bw = blockDim.x;
  Dims b_dims{bh, bw};
  auto lattice = lattices[blockIdx.x];
  extern __shared__ MinMax shm[];
  GridView<MinMax> c_minmax{shm, b_dims};
  GridView<MinMax> r_minmax = {shm + b_dims.vol(), b_dims};
  int32_t tr = threadIdx.y;
  int32_t tc = threadIdx.x;
  c_minmax(tr, tc) = MinMax{lattice.dims.width, 0};
  r_minmax(tr, tc) = MinMax{lattice.dims.width, 0};
  GRID_FOR(1, 1, lattice.dims.height - 1, lattice.dims.width - 1) {
    if (lattice(r, c).isOccupied()) {
      c_minmax(tr, tc).min = min(c_minmax(tr, tc).min, c);
      c_minmax(tr, tc).max = max(c_minmax(tr, tc).max, c);
      r_minmax(tr, tc).min = min(r_minmax(tr, tc).min, r);
      r_minmax(tr, tc).max = max(r_minmax(tr, tc).max, r);
    }
  }
  __syncthreads();
  int32_t tci = tc;
  int32_t tri = tr;
  for (int32_t d = 1; d < b_dims.width; d *= 2) {
    if (tci % 2 == 0 && tri % 2 == 0) {
      reduce2d_min(c_minmax, tr, tc, d);
      reduce2d_min(r_minmax, tr, tc, d);
      reduce2d_max(c_minmax, tr, tc, d);
      reduce2d_max(r_minmax, tr, tc, d);
      tci /= 2;
      tri /= 2;
    }
    __syncthreads();
  }
  if (tr == 0 && tc == 0) {
    const auto lt_c = max(1, c_minmax(0, 0).min - 1);
    const auto lt_r = max(1, r_minmax(0, 0).min - 1);
    const auto rb_c = min(lattice.dims.width - 2, c_minmax(0, 0).max + 1);
    const auto rb_r = min(lattice.dims.height - 2, r_minmax(0, 0).max + 1);
    rois[blockIdx.x] = ROI{Coords{lt_r, lt_c}, Dims{rb_r - lt_r + 1, rb_c - lt_c + 1}};
  }
}

void findTumorsBoundaries(const GridView<Site> *lattices, ROI *rois, uint32_t batch_size) {
  findBoundariesKernel
    <<<batch_size, dim3(CuBlockDimX, CuBlockDimY), CuBlockDimX*CuBlockDimY*2*sizeof(MinMax)>>>
    (lattices, rois);
  KERNEL_DEBUG("tumor boundaries")
}

}  // namespace emt6ro
