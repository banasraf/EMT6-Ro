#ifndef SRC_EMT6RO_COMMON_SUBSTRATES_H_
#define SRC_EMT6RO_COMMON_SUBSTRATES_H_

#include <cuda_runtime.h>
#include <type_traits>

constexpr bool DEBUG = false;

#define KERNEL_DEBUG(name) \
if (DEBUG) { \
  cudaDeviceSynchronize(); \
  auto code = cudaPeekAtLastError();\
  if (code != cudaSuccess) { \
    std::cout << "error in " << name << ": " << cudaGetErrorString(code) << std::endl; \
    exit(-1); \
  } \
}

namespace emt6ro {

struct Substrates {
  float cho;  // glucose
  float ox;   // oxygen
  float gi;   // metabolism by-products

  __host__ __device__ inline Substrates &operator +=(const Substrates &rhs) {
    cho += rhs.cho;
    ox += rhs.ox;
    gi += rhs.gi;
    return *this;
  }

  __host__ __device__ inline Substrates &operator -=(const Substrates &rhs) {
    cho -= rhs.cho;
    ox -= rhs.ox;
    gi -= rhs.gi;
    return *this;
  }

  __host__ __device__ inline Substrates &operator *=(const Substrates &rhs) {
    cho *= rhs.cho;
    ox *= rhs.ox;
    gi *= rhs.gi;
    return *this;
  }

  __host__ __device__ inline Substrates &operator *=(float f) {
    cho *= f;
    ox *= f;
    gi *= f;
    return *this;
  }

  __host__ __device__ inline Substrates &operator /=(float f) {
    cho /= f;
    ox /= f;
    gi /= f;
    return *this;
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

  __host__ __device__ inline Substrates operator /(float f) const {
    auto c = *this;
    return c /= f;
  }

//  __host__ __device__ Substrates &operator =(const Substrates &rhs) = default;

};

static_assert(std::is_pod<Substrates>::value, "");

}  // namespace emt6ro

#endif  // SRC_EMT6RO_COMMON_SUBSTRATES_H_
