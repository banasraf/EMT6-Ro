#include "emt6ro/common/random-engine.h"

namespace emt6ro {

namespace detail {

__global__ void initializeState(curandState_t *state, const uint32_t *seeds, size_t size) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= size) return;
  curand_init(seeds[i], 0, 0, &state[i]);
}

void init(curandState_t *state_data, const uint32_t *seeds, size_t size, cudaStream_t stream) {
  size_t block_size = (size > 1024) ? 1024 : size;
  size_t grid_size = (size + block_size - 1) / block_size;
  detail::initializeState<<<grid_size, block_size, 0, stream>>>(state_data, seeds, size);
}

}  // namespace detail

CuRandEngineState::CuRandEngineState(size_t size, const uint32_t* seeds) : state_(size) {
  init(seeds);
}

CuRandEngineState::CuRandEngineState(size_t size): state_(size) {}

void CuRandEngineState::init(const uint32_t *seeds, cudaStream_t stream) {
  detail::init(state_.data(), seeds, state_.size(), stream);
}

__device__ float CuRandEngine::uniform() {
  return curand_uniform(state);
}

__device__ float CuRandEngine::normal(const Parameters::NormalDistribution& params) {
  return  params.stddev * curand_normal(state) + params.mean;
}

}  // namespace emt6ro
