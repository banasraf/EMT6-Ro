#ifndef EMT6RO_COMMON_RANDOM_ENGINE_H_
#define EMT6RO_COMMON_RANDOM_ENGINE_H_

#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <random>
#include "emt6ro/parameters/parameters.h"
#include "emt6ro/common/device-buffer.h"

namespace emt6ro {

class CuRandEngineState {
 public:
  CuRandEngineState(size_t size, const uint32_t *seeds);

  explicit CuRandEngineState(size_t size);

  /**
   * Initialize the random engine state.
   * @param seeds - pointer to device data with random seeds
   * @param stream - cuda stream
   */
  void init(const uint32_t *seeds, cudaStream_t stream = nullptr);

  inline const curandState_t *states() const {
    return state_.data();
  }

  inline curandState_t *states() {
    return state_.data();
  }

 private:
  device::buffer<curandState_t> state_;
};

class CuRandEngine {
 public:
  explicit __host__ __device__ CuRandEngine(curandState_t *state): state(state) {}

  __device__ float uniform();

  __device__ float normal(const Parameters::NormalDistribution &params);

 private:
  curandState_t *state;
};

class HostRandEngine {
 public:
  explicit HostRandEngine(uint32_t seed): gen{seed} {}

  float uniform();

  float normal(const Parameters::NormalDistribution &params);

 private:
  std::mt19937 gen;
};

}  // namespace emt6ro
#endif  // EMT6RO_COMMON_RANDOM_ENGINE_H_
