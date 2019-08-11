#ifndef SRC_EMT6RO_RANDOM_ENGINE_H
#define SRC_EMT6RO_RANDOM_ENGINE_H

#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include "emt6ro/parameters/parameters.h"
#include "emt6ro/common/device-buffer.h"

namespace emt6ro {

class CuRandEngineState {
 public:
  CuRandEngineState(size_t size, uint32_t *seeds);

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
  explicit __host__ __device__ CuRandEngine(curandState_t *state): state(state) {};

  __device__ float uniform();

  __device__ float normal(const Parameters::NormalDistribution &params);

 private:
  curandState_t *state;
};

}  // namespace emt6ro
#endif  // SRC_EMT6RO_RANDOM_ENGINE_H