#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <algorithm>
#include "emt6ro/diffusion/diffusion.h"
#include "emt6ro/common/debug.h"
#include "emt6ro/common/cuda-utils.h"

namespace emt6ro {

__device__ void fillBorderMask(GridView<uint8_t> mask, int32_t max_dist) {
  float midr = mask.dims.height / 2.f;
  float midc = mask.dims.width / 2.f;
  auto dist = [=](int32_t r, int32_t c) {
    float fr = r, fc = c;
    return sqrtf((fr-midr)*(fr-midr) + (fc-midc)*(fc-midc));
  };
  GRID_FOR(0, 0, mask.dims.height, mask.dims.width) {
    if (dist(r+1, c+1) >= max_dist) {
      mask(r, c) = 1;
    } else {
      mask(r, c) = 0;
    }
  }
}

__device__ int32_t distance(Coords a, Coords b) {
  auto dist2_f = static_cast<float>((a.r-b.r)*(a.r-b.r) + (a.c-b.c)*(a.c-b.c));
  auto dist_f = sqrtf(dist2_f);
  return static_cast<int32_t>(ceilf(dist_f));
}

__device__ uint64_t coords_minmax(uint64_t lhs, uint64_t rhs) {
  auto lhs_mm = Coords2::decode(lhs);
  auto rhs_mm = Coords2::decode(rhs);
  Coords2 result;
  result[0].r = min(lhs_mm[0].r, rhs_mm[0].r);
  result[0].c = min(lhs_mm[0].c, rhs_mm[0].c);
  result[1].r = max(lhs_mm[1].r, rhs_mm[1].r);
  result[1].c = max(lhs_mm[1].c, rhs_mm[1].c);
  return result.encode();
}

template <typename F>
__device__ Coords2 reduceCoords(GridView<const Site> lattice,
                                F &op, Coords2 zero,
                                void *shared_mem) {
  uint64_t coords = zero.encode();
  GRID_FOR(0, 0, lattice.dims.height, lattice.dims.width) {
    if (lattice(r, c).isOccupied()) {
      coords = op(coords, Coords2({Coords(r, c), Coords(r, c)}).encode());
    }
  }
  
  auto *shm = static_cast<uint64_t*>(shared_mem);
  return Coords2::decode(block_reduce(coords, shm, op));
}


__device__ ROI findROI(GridView<const Site> lattice, uint8_t *b_mask_mem,
                       void *shared_mem) {
  Coords2 zero;
  zero[0] = Coords(lattice.dims.height, lattice.dims.width);
  zero[1] = Coords(0, 0);
  Coords2 min_max = reduceCoords(lattice, coords_minmax, zero, shared_mem);
  Coords min_ = min_max[0];
  Coords max_ = min_max[1];
  if (max_.r == 0 && max_.c == 0) return ROI{Coords(0, 0), Dims(0, 0)};
  Coords mid{};
  mid.r = min_.r + (max_.r - min_.r + 1) / 2;
  mid.c = min_.c + (max_.c - min_.c + 1) / 2;
  auto collect_dist = [=] (int32_t d, Coords coords) {
    auto dist = distance(mid, coords);
    return max(d, dist);
  };
  int dist = 0;
  GRID_FOR(min_.r, min_.c, max_.r + 1, max_.c + 1) {
    if (lattice(r, c).isOccupied())
      dist = collect_dist(dist, Coords(r, c));
  }
  auto *sh_d = static_cast<int*>(shared_mem);
  dist = block_reduce(dist, sh_d, [](int a, int b) {return max(a, b);});
  int16_t sub_r = (mid.r <= dist) ? 1 : mid.r - dist;
  int16_t sub_c = (mid.c <= dist) ? 1 : mid.c - dist;
  int16_t sub_w =
      (mid.c + dist >= lattice.dims.width-1) ? lattice.dims.width-1 - sub_c
                                           : mid.c + dist - sub_c + 1;
  int16_t sub_h =
      (mid.r + dist >= lattice.dims.height-1) ? lattice.dims.height-1 - sub_r
                                            : mid.r + dist - sub_r + 1;
  ROI roi{Coords(sub_r, sub_c), Dims(sub_h, sub_w)};
  GridView<uint8_t> b_mask{b_mask_mem, Dims(sub_h + 2, sub_w + 2)};
  fillBorderMask(b_mask, dist);
  return roi;
}

__global__ void findROIsKernel(ROI *rois, uint8_t *border_masks, const GridView<Site> *lattices) {
  auto lattice = lattices[blockIdx.x];
  uint8_t *border_mask = border_masks + blockIdx.x * lattice.dims.vol();
  extern __shared__ char shared_mem[];
  auto roi = findROI(lattice, border_mask, shared_mem);
  if (threadIdx.x == 0 && threadIdx.y == 0)
    rois[blockIdx.x] = roi;
}

void findROIs(ROI *rois, uint8_t *border_masks, const GridView<Site> *lattices,
              int32_t batch_size, cudaStream_t stream) {
  auto shared_size = sizeof(uint64_t) * CuBlockDimY * CuBlockDimX / 32;
  findROIsKernel<<<batch_size, dim3(CuBlockDimX, CuBlockDimY), shared_size, stream>>>
      (rois, border_masks, lattices);
  KERNEL_DEBUG("findROIs kernel")
}

constexpr float HS = 2 * M_SQRT2;
constexpr float f = 4. + HS;

__device__ Substrates diffusion_differential(const GridView<Substrates> &lattice,
                                             int16_t r, int16_t c,
                                             Substrates coeffs) {
  
  Substrates result = lattice(r - 1, c) + lattice(r, c - 1) +
                      lattice(r + 1, c) + lattice(r, c + 1);
  result += (lattice(r - 1, c - 1) + lattice(r - 1, c + 1)  +
             lattice(r + 1, c - 1) + lattice(r + 1, c + 1)) *
            M_SQRT1_2;
  result -= lattice(r, c) * f;
  result *= coeffs;
  return result;
}

__global__ void diffusionKernel(GridView<Site> *lattices, const ROI *rois,
                                const uint8_t *border_masks,
                                Parameters::Diffusion params,
                                Substrates external_levels, int16_t steps) {
  extern __shared__ Substrates tmp_mem[];
  auto lattice = lattices[blockIdx.x];
  auto roi = rois[blockIdx.x];
  Dims bordered_dims(roi.dims.height + 4, roi.dims.width + 4);
  GridView<Substrates> tmp_grid{tmp_mem, bordered_dims};
  Substrates diff[SitesPerThread/4];
  Coords sites[SitesPerThread/4];
  int16_t nsites=0;
  GridView<const uint8_t> b_mask{border_masks + lattice.dims.vol() * blockIdx.x, Dims(roi.dims.height + 2, roi.dims.width + 2)};
  GRID_FOR(roi.origin.r-2, roi.origin.c-2,
           roi.origin.r + roi.dims.height+2, roi.origin.c + roi.dims.width+2) {
    auto dr = r - roi.origin.r + 2;
    auto dc = c - roi.origin.c + 2;
    if (dr == 0 || dr == bordered_dims.height-1 || dc == 0 || dc == bordered_dims.width-1)
      tmp_grid(dr, dc) = {0.f, 0.f, 0.f};
    else if (b_mask(dr-1, dc-1))
      tmp_grid(dr, dc) = external_levels;
    else {
      tmp_grid(dr, dc) = lattice(r, c).substrates;
      sites[nsites++] = Coords(dr, dc);
    }
  }
  auto coeffs = (params.coeffs * params.time_step * HS) / f;
  __syncthreads();
  for (int16_t s = 0; s < steps - 1; ++s) {
    for (int16_t i = 0; i < nsites; ++i) 
      diff[i] = diffusion_differential(tmp_grid, sites[i].r, sites[i].c, coeffs);
    __syncthreads();
    for (int16_t i = 0; i < nsites; ++i)
      tmp_grid(sites[i].r, sites[i].c) += diff[i];
    __syncthreads();
  }
  uint8_t subi = 0;
  GRID_FOR(0, 0, roi.dims.height, roi.dims.width) {
    diff[subi] = diffusion_differential(tmp_grid, r+2, c+2, coeffs);
    ++subi;
  }
  __syncthreads();
  subi = 0;
  GRID_FOR(0, 0, roi.dims.height, roi.dims.width) {
    tmp_grid(r+2, c+2) += diff[subi];
    ++subi;
  }
  __syncthreads();
  GRID_FOR(roi.origin.r, roi.origin.c,
           roi.origin.r + roi.dims.height, roi.origin.c + roi.dims.width) {
      auto dr = r - roi.origin.r + 2;
      auto dc = c - roi.origin.c + 2;
      lattice(r, c).substrates = tmp_grid(dr, dc);
  }
}

void batchDiffusion(GridView<Site> *lattices, const ROI *rois, const uint8_t *border_masks,
                    const Parameters::Diffusion &params, Substrates external_levels, int16_t steps,
                    Dims dims, int32_t batch_size, cudaStream_t stream) {
  auto shared_mem_size = sizeof(Substrates) * Dims(dims.height+4, dims.width+4).vol();
  diffusionKernel<<<batch_size, dim3(CuBlockDimX*2, CuBlockDimY*2), shared_mem_size, stream>>>
    (lattices, rois, border_masks, params, external_levels, steps);
  KERNEL_DEBUG("diffusion kernel")
}

}  // namespace emt6ro
