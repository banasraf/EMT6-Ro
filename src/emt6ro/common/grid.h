#ifndef EMT6RO_COMMON_GRID_H_
#define EMT6RO_COMMON_GRID_H_
#include <cuda_runtime.h>
#include <cstdint>
#include <memory>

namespace emt6ro {

struct Dims {
  int32_t height;
  int32_t width;

  inline __host__ __device__ int32_t vol() const {
    return height * width;
  }
};

struct Coords {
  int32_t r;
  int32_t c;
};

/// Region of interest
struct ROI {
  Coords origin;
  Dims dims;
};

template <typename T>
struct GridView {
  T *data;
  Dims dims;

  inline __host__ __device__  const T& operator()(int32_t r, int32_t c) const {
    return data[r * dims.width + c];
  }

  inline __host__ __device__ T& operator()(int32_t r, int32_t c) {
    return data[r * dims.width + c];
  }

  inline __host__ __device__ const T& operator()(const Coords &coords) const {
    return data[coords.r * dims.width + coords.c];
  }

  inline __host__ __device__ T& operator()(const Coords &coords) {
    return data[coords.r * dims.width + coords.c];
  }
};

template <typename T>
class HostGrid {
  std::unique_ptr<T[]> data_;
  GridView<T> view_;

 public:
  explicit HostGrid(Dims dims): data_(new T[dims.vol()]), view_{data_.get(), dims} {}

  GridView<T> view() {
    return view_;
  }
};

}  // namespace emt6ro

static constexpr uint32_t CuBlockDimX = 32;
static constexpr uint32_t CuBlockDimY = 32;

#define GRID_FOR(START_R, START_C, END_R, END_C) \
for (int32_t r = threadIdx.y + START_R; r < END_R; r += blockDim.y) \
  for (int32_t c = threadIdx.x + START_C; c < END_C; c += blockDim.x)

#endif  // EMT6RO_COMMON_GRID_H_
