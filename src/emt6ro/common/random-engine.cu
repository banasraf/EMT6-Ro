#include "emt6ro/common/random-engine.h"

namespace emt6ro {

namespace detail {

__global__ void initializeState(curandState_t *state, uint32_t *seeds, size_t size) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  curand_init(seeds[i], 0, 0, &state[i]);
}

}  // namespace detail

CuRandEngineState::CuRandEngineState(size_t size, uint32_t* seeds) : state_(size) {
  size_t block_size = (size > 1024) ? 1024 : size;
  size_t grid_size = (size + block_size - 1) / block_size;
  detail::initializeState<<<grid_size, block_size>>>(state_.data(), seeds, size);
}

__device__ float CuRandEngine::uniform() {
  return curand_uniform(state);
}

__device__ float CuRandEngine::normal(const Parameters::NormalDistribution& params) {
  return  params.stddev * curand_normal(state) + params.mean;
}

}  // namespace emt6ro