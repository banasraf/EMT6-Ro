#ifndef SRC_EMT6RO_COMMON_SUBSTRATES_H_
#define SRC_EMT6RO_COMMON_SUBSTRATES_H_

#include <cuda_runtime.h>

namespace emt6ro {

struct Substrates {
  float cho;  // glucose
  float ox;   // oxygen
  float gi;   // metabolism by-products

  __host__ __device__ inline void operator +=(const Substrates &rhs) {
    cho += rhs.cho;
    ox += rhs.ox;
    gi += rhs.gi;
  }
};

}  // namespace emt6ro

#endif  // SRC_EMT6RO_COMMON_SUBSTRATES_H_
