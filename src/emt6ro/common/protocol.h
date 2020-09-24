#ifndef EMT6RO_COMMON_PROTOCOL_H_
#define EMT6RO_COMMON_PROTOCOL_H_

#include <cuda_runtime.h>
#include <cstdint>

namespace emt6ro {
struct Protocol {
  __host__ __device__ inline float getDose(uint32_t step) const {
    if (step < length_ && step % step_resolution_ == 0) {
      return data_[step / step_resolution_];
    }
    return 0;
  }

  __host__ __device__ inline float &closestDose(uint32_t step) {
    return data_[step / step_resolution_];
  }

  __device__ void reset();

  uint32_t step_resolution_;
  uint32_t length_;
  float *data_;
};
}  // namespace emt6ro
#endif  // EMT6RO_COMMON_PROTOCOL_H_
