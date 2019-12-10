#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <algorithm>
#include "emt6ro/diffusion/diffusion.h"
#include "emt6ro/common/debug.h"

namespace emt6ro {

template <typename Result, typename Reduction>
__device__ void reduce2d(GridView<Result> &grid, int32_t r, int32_t c, int32_t d,
                         const Reduction &reduction) {
  grid(r, c) = reduction(grid(r, c), grid(r, c + d));
  grid(r + d, c) = reduction(grid(r + d, c), grid(r + d, c + d));
  grid(r, c)= reduction(grid(r, c), grid(r + d, c));
}

template <typename Result, typename Collect, typename Reduce>
__device__ void coordsReduction(GridView<Result> shared_view, const GridView<Site> &lattice,
                                Coords origin, Dims sub_dims,
                                const Collect &collect, const Reduce &reduce) {
  Dims b_dims = shared_view.dims;
  GRID_FOR(origin.r, origin.c, origin.r + sub_dims.height, origin.c + sub_dims.width) {
    if (lattice(r, c).isOccupied())
      shared_view(threadIdx.y, threadIdx.x)
        = collect(shared_view(threadIdx.y, threadIdx.x), Coords{r, c});
  }
  __syncthreads();
  int32_t tci = threadIdx.x;
  int32_t tri = threadIdx.y;
  for (int32_t d = 1; d < b_dims.width; d *= 2) {
    if (tci % 2 == 0 && tri % 2 == 0) {
      reduce2d(shared_view, threadIdx.y, threadIdx.x, d, reduce);
      tci /= 2;
      tri /= 2;
    }
  __syncthreads();
  }
}

struct CoordsMinMax {
  Coords min;
  Coords max;

  __host__ __device__ CoordsMinMax(Dims dims) {
    max.r = 0;
    max.c = 0;
    min.r = dims.height;
    min.c = dims.width;
  }

  CoordsMinMax() = default;
};

__device__ CoordsMinMax collectCoords(const CoordsMinMax &min_max, Coords coords) {
  CoordsMinMax result;
  result.min.r = min(min_max.min.r, coords.r);
  result.min.c = min(min_max.min.c, coords.c);
  result.max.r = max(min_max.max.r, coords.r);
  result.max.c = max(min_max.max.c, coords.c);
  return result;
}

__device__ CoordsMinMax reduceCoords(const CoordsMinMax &lhs, const CoordsMinMax &rhs) {
  CoordsMinMax result;
  result.min.r = min(lhs.min.r, rhs.min.r);
  result.min.c = min(lhs.min.c, rhs.min.c);
  result.max.r = max(lhs.max.r, rhs.max.r);
  result.max.c = max(lhs.max.c, rhs.max.c);
  return result;
}

__device__ int32_t maxi(int32_t rhs, int32_t lhs) {
  return max(rhs, lhs);
}

__device__ int32_t distance(Coords a, Coords b) {
  auto dist2_f = static_cast<float>((a.r-b.r)*(a.r-b.r) + (a.c-b.c)*(a.c-b.c));
  auto dist_f = sqrtf(dist2_f);
  return static_cast<int32_t>(ceilf(dist_f));
}

__device__ void fillBorderMask(uint8_t *border_mask, ROI roi, Coords mid, int32_t max_dist) {
  GridView<uint8_t> mask{border_mask, {roi.dims.height + 2, roi.dims.width + 2}};
  float midr = roi.dims.height / 2.f;
  float midc = roi.dims.width / 2.f;
  auto dist = [=](int32_t r, int32_t c) {
    float fr = r, fc = c;
    return sqrtf((fr-midr)*(fr-midr) + (fc-midc)*(fc-midc));
  };
  GRID_FOR(0, 0, roi.dims.height+2, roi.dims.width+2) {
    if (dist(r, c) >= max_dist) {
      mask(r, c) = 1;
    } else {
      mask(r, c) = 0;
    }
  }
}

__global__ void findROIsKernel(ROI *rois, uint8_t *border_masks, const GridView<Site> *lattices) {
  auto &roi = rois[blockIdx.x];
  auto lattice = lattices[blockIdx.x];
  extern  __shared__ CoordsMinMax shared_mem[];
  int32_t bh = blockDim.y;
  int32_t bw = blockDim.x;
  Dims b_dims{bh, bw};
  auto shared_dist = reinterpret_cast<int32_t*>(shared_mem + b_dims.vol());
  GridView<CoordsMinMax> shared_view{shared_mem, b_dims};
  shared_view(threadIdx.y, threadIdx.x) = CoordsMinMax(lattice.dims);
  __syncthreads();
  coordsReduction(shared_view, lattice, {1, 1}, {lattice.dims.height-2, lattice.dims.width-2},
                  collectCoords, reduceCoords);
  Coords min_ = shared_mem[0].min;
  Coords max_ = shared_mem[0].max;
  Coords mid{};
  mid.r =
      min_.r + __float2int_rn(0.5 * (max_.r - min_.r));
  mid.c =
      min_.c + __float2int_rn(0.5 * (max_.c - min_.c));
  auto collect_dist = [=] (int32_t d, Coords coords) {
    auto dist = distance(mid, coords);
    return max(d, dist);
  };
  GridView<int32_t> dist_view{shared_dist, b_dims};
  dist_view(threadIdx.y, threadIdx.x) = 0;
  __syncthreads();
  coordsReduction(dist_view, lattice, min_,
                  {max_.r - min_.r + 1, max_.c - min_.c + 1},
                  collect_dist, maxi);
  int32_t max_dist = dist_view(0, 0);
  int32_t sub_r = (mid.r <= max_dist) ? 1 : mid.r - max_dist;
  int32_t sub_c = (mid.c <= max_dist) ? 1 : mid.c - max_dist;
  int32_t sub_w =
      (mid.c + max_dist >= lattice.dims.width-1) ? lattice.dims.width-1 - sub_c
                                           : mid.c + max_dist - sub_c + 1;
  int32_t sub_h =
      (mid.r + max_dist >= lattice.dims.height-1) ? lattice.dims.height-1 - sub_r
                                            : mid.r + max_dist - sub_r + 1;
  if (threadIdx.x == 0 || threadIdx.y == 0)
    roi = {{sub_r, sub_c}, {sub_h, sub_w}};
  __syncthreads();
  fillBorderMask(border_masks + lattice.dims.vol() * blockIdx.x, roi, mid, max_dist);
}

void findROIs(ROI *rois, uint8_t *border_masks, const GridView<Site> *lattices,
              int32_t batch_size, cudaStream_t stream) {
  auto shared_size = (sizeof(CoordsMinMax) + sizeof(int32_t)) * CuBlockDimY * CuBlockDimX;
  findROIsKernel<<<batch_size, dim3(CuBlockDimX, CuBlockDimY), shared_size, stream>>>
      (rois, border_masks, lattices);
  KERNEL_DEBUG("findROIs kernel")
}

__device__ Substrates diffusion_differential(const GridView<Substrates> &lattice,
                                             int32_t r, int32_t c,
                                             const Parameters::Diffusion &params) {
  constexpr float HS = 2 * M_SQRT2f32;
  constexpr float f = 4. + HS;
  Substrates result = lattice(r - 1, c) + lattice(r, c - 1) +
                      lattice(r + 1, c) + lattice(r, c + 1);
  result += (lattice(r - 1, c - 1)+ lattice(r - 1, c + 1) +
             lattice(r + 1, c - 1) + lattice(r + 1, c + 1)) *
            M_SQRT1_2f32;
  result -= lattice(r, c) * f;
  result *= (params.coeffs * params.time_step * HS) / f;
  return result;
}

__global__ void diffusionKernel(GridView<Site> *lattices, const ROI *rois,
                                const uint8_t *border_masks,
                                Parameters::Diffusion params,
                                Substrates external_levels, int32_t steps) {
  extern __shared__ Substrates tmp_mem[];
  auto lattice = lattices[blockIdx.x];
  auto roi = rois[blockIdx.x];
  Dims bordered_dims{roi.dims.height + 2, roi.dims.width + 2};
  GridView<Substrates> tmp_grid{tmp_mem, bordered_dims};
  Substrates diff[4];
  GridView<const uint8_t> b_mask{border_masks + lattice.dims.vol() * blockIdx.x, bordered_dims};
  GRID_FOR(roi.origin.r-1, roi.origin.c-1,
           roi.origin.r + roi.dims.height+1, roi.origin.c + roi.dims.width+1) {
    auto dr = r - roi.origin.r + 1;
    auto dc = c - roi.origin.c + 1;
    if (b_mask(dr, dc))
      tmp_grid(dr, dc) = external_levels;
    else
      tmp_grid(dr, dc) = lattice(r, c).substrates;
  }
  __syncthreads();
  uint8_t subi = 0;
  for (int32_t i = 0; i < steps; ++i) {
    subi = 0;
    GRID_FOR(1, 1, roi.dims.height, roi.dims.width) {
      if (!b_mask(r, c)) {
        diff[subi] = diffusion_differential(tmp_grid, r, c, params);
      }
      ++subi;
    }
    __syncthreads();
    subi = 0;
    GRID_FOR(1, 1, roi.dims.height, roi.dims.width) {
      if (!b_mask(r, c)) {
        tmp_grid(r, c) += diff[subi];
      }
      ++subi;
    }
    __syncthreads();
  }
  GRID_FOR(roi.origin.r, roi.origin.c,
           roi.origin.r + roi.dims.height, roi.origin.c + roi.dims.width) {
      auto dr = r - roi.origin.r + 1;
      auto dc = c - roi.origin.c + 1;
      lattice(r, c).substrates = tmp_grid(dr, dc);
  }
}

void batchDiffusion(GridView<Site> *lattices, const ROI *rois, const uint8_t *border_masks,
                    const Parameters::Diffusion &params, Substrates external_levels, int32_t steps,
                    Dims dims, int32_t batch_size, cudaStream_t stream) {
  auto shared_mem_size = sizeof(Substrates) * Dims{dims.height + 2, dims.width + 2}.vol();
  diffusionKernel<<<batch_size, dim3(CuBlockDimX, CuBlockDimY), shared_mem_size, stream>>>
    (lattices, rois, border_masks, params, external_levels, steps);
  KERNEL_DEBUG("diffusion kernel")
}

}  // namespace emt6ro
