#ifndef SRC_EMT6RO_COMMON_GRID_H_
#define SRC_EMT6RO_COMMON_GRID_H_
#include <cuda_runtime.h>
#include <cstdint>

namespace emt6ro {

struct Dims {
  uint32_t height;
  uint32_t width;

  inline __host__ __device__ uint32_t vol() const {
    return height * width;
  }
};

struct Coords {
  uint32_t r;
  uint32_t c;
};

template <typename T>
struct GridView {
  T *data;
  Dims dims;

  inline __host__ __device__  const T& operator()(uint32_t r, uint32_t c) const {
    return data[r * dims.width + c];
  }

  inline __host__ __device__ T& operator()(uint32_t r, uint32_t c) {
    return data[r * dims.width + c];
  }

  inline __host__ __device__ const T& operator()(const Coords &coords) const {
    return (*this)(coords.r, coords.c);
  }

  inline __host__ __device__ T& operator()(const Coords &coords) {
    return (*this)(coords.r, coords.c);
  }
};

}  // namespace emt6ro

#define GRID_FOR(START_R, START_C, END_R, END_C) \
for (uint32_t r = threadIdx.y + START_R; r < END_R; r += blockDim.y) \
  for (uint32_t c = threadIdx.x + START_C; c < END_C; c += blockDim.x)

#endif  // SRC_EMT6RO_COMMON_GRID_H_
