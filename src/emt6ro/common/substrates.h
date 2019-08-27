#ifndef SRC_EMT6RO_COMMON_SUBSTRATES_H_
#define SRC_EMT6RO_COMMON_SUBSTRATES_H_

#include <cuda_runtime.h>

namespace emt6ro {

struct Substrates {
  float cho;  // glucose
  float ox;   // oxygen
  float gi;   // metabolism by-products

  __host__ __device__ inline Substrates &operator +=(const Substrates &rhs) {
    cho += rhs.cho;
    ox += rhs.ox;
    gi += rhs.gi;
  }

  __host__ __device__ inline Substrates &operator -=(const Substrates &rhs) {
    cho -= rhs.cho;
    ox -= rhs.ox;
    gi -= rhs.gi;
  }

  __host__ __device__ inline Substrates &operator *=(const Substrates &rhs) {
    cho *= rhs.cho;
    ox *= rhs.ox;
    gi *= rhs.gi;
  }

  __host__ __device__ inline Substrates &operator *=(float f) {
    cho *= f;
    ox *= f;
    gi *= f;
  }

  __host__ __device__ inline Substrates operator +(const Substrates &rhs) const {
    auto c = *this;
    return c += rhs;
  }

  __host__ __device__ inline Substrates operator -(const Substrates &rhs) const {
    auto c = *this;
    return c-= rhs;
  }

  __host__ __device__ inline Substrates operator *(const Substrates &rhs) const {
    auto c = *this;
    return c *= rhs;
  }

  __host__ __device__ inline Substrates operator *(float f) const {
    auto c = *this;
    return c *= f;
  }
};

}  // namespace emt6ro

#endif  // SRC_EMT6RO_COMMON_SUBSTRATES_H_
