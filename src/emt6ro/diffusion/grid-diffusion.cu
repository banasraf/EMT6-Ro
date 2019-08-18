#include "emt6ro/diffusion/grid-diffusion.h"
#include <cuda_runtime.h>
#include <cmath>

namespace emt6ro {

__host__ __device__ double paramDiffusion(float val, float coeff, float tau,
                                 float ortho_sum, float diag_sum) {
  constexpr float HS = 2 * M_SQRT2f32;
  constexpr float f = 4. + HS;
  return (coeff*tau*HS)/f * (ortho_sum + M_SQRT1_2f32*diag_sum - f*val) + val;
}

__device__ void diffuse(const GridView<float> &input, GridView<float> &output,
                        float coeff, float time_step) {
  const auto h = input.dims.height;
  const auto w = input.dims.width;
  GRID_FOR(1, 1, h - 1, w - 1) {
    auto diag = input(r - 1, c - 1) +
        input(r - 1, c + 1) +
        input(r + 1, c - 1) +
        input(r + 1, c + 1);
    auto ortho = input(r - 1, c) +
        input(r, c - 1) +
        input(r + 1, c) +
        input(r, c + 1);

    output(r, c) = paramDiffusion(input(r, c), coeff, time_step, ortho, diag);
  }
}

namespace {
__global__ void diffuseKernel(float *data, Dims dims,
                              float coeff, float time_step, uint32_t steps) {
  extern __shared__ float tmp[];
  GridView<float> data_view{data + blockIdx.x * dims.vol(), dims};
  GridView<float> tmp_view{tmp, dims};
  GRID_FOR(0, 0, dims.height, dims.width) {
    tmp_view(r, c) = data_view(r, c);
  }
  for (uint32_t i = 0; i < steps; i += 2) {
    diffuse(data_view, tmp_view, coeff, time_step);
    __syncthreads();
    diffuse(tmp_view, data_view, coeff, time_step);
    __syncthreads();
  }
}
}  // namespace

void batchDiffuse(float *data, Dims dims, size_t batch_size,
                  float coeff, float time_step, uint32_t steps) {
  diffuseKernel<<<batch_size, dim3(32, 32), dims.vol() * sizeof(float)>>>
    (data, dims, coeff, time_step, steps);
}

namespace {

struct SelectCho {
  static inline __device__ const float &select(const Site &site) {
    return site.substrates.cho;
  }

  static inline __device__ float &select(Site &site) {
    return site.substrates.cho;
  }
};

struct SelectOx {
  static inline __device__ const float &select(const Site &site) {
    return site.substrates.ox;
  }

  static inline __device__ float &select(Site &site) {
    return site.substrates.ox;
  }
};

struct SelectGi {
  static inline __device__ const float &select(const Site &site) {
    return site.substrates.gi;
  }

  static inline __device__ float &select(Site &site) {
    return site.substrates.gi;
  }
};

template <typename Selector>
__global__ void copySubstrate(float *substrate, const Site *sites, uint32_t size) {
  const auto x = blockDim.x * blockIdx.x + threadIdx.x;
  if (x < size) {
    substrate[x] = Selector::select(sites[x]);
  }
}

template <typename Selector>
__global__ void copySubstrateBack(Site *sites, const float *substrate, uint32_t size) {
  const auto x = blockDim.x * blockIdx.x + threadIdx.x;
  if (x < size) {
    Selector::select(sites[x]) = substrate[x];
  }
}

}  // namespace

void copySubstrates(float* cho, float* ox, float* gi, const Site* sites, Dims dims,
                    uint32_t batch_size) {
  const auto size = dims.vol() * batch_size;
  const auto blocks = (size + 1023) / 1024;
  copySubstrate<SelectCho><<<blocks, 1024>>>(cho, sites, size);
  copySubstrate<SelectOx><<<blocks, 1024>>>(ox, sites, size);
  copySubstrate<SelectGi><<<blocks, 1024>>>(gi, sites, size);
}
void copySubstratesBack(Site *sites, const float *cho, const float *ox, const float *gi, Dims dims,
                        uint32_t batch_size) {
  const auto size = dims.vol() * batch_size;
  const auto blocks = (size + 1023) / 1024;
  copySubstrateBack<SelectCho><<<blocks, 1024>>>(sites, cho, size);
  copySubstrateBack<SelectOx><<<blocks, 1024>>>(sites, ox, size);
  copySubstrateBack<SelectGi><<<blocks, 1024>>>(sites, gi, size);
}

}  // namespace emt6ro
