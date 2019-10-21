#ifndef EMT6RO_SIMULATION_PROTOCOL_H
#define EMT6RO_SIMULATION_PROTOCOL_H

#include <cstdint>
#include <cuda_runtime.h>

namespace emt6ro {
struct Protocol {
  __host__ __device__ inline float getDose(uint32_t step) const {
    if (step < length_ && step % step_resolution_ == 0) {
      return data_[step / step_resolution_];
    }
    return 0;
  }

  uint32_t step_resolution_;
  uint32_t length_;
  float *data_;
};
}  // namespace emt6ro
#endif  // EMT6RO_SIMULATION_PROTOCOL_H
