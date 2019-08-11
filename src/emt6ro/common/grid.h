#ifndef SRC_EMT6RO_COMMON_GRID_H_
#define SRC_EMT6RO_COMMON_GRID_H_
#include <cuda_runtime.h>
#include <cstdint>

namespace emt6ro {

struct Dims {
  uint32_t height;
  uint32_t width;

  inline __host__ __device__ uint32_t vol() {
    return height * width;
  }
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
};

}  // namespace emt6ro

#define GRID_FOR(H, W) \
for (uint32_t r = threadIdx.y + 1; r < H - 1; r += blockDim.y) \
  for (uint32_t c = threadIdx.x + 1; c < W - 1; c += blockDim.x)

#endif  // SRC_EMT6RO_COMMON_GRID_H_
