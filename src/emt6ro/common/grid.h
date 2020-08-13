#ifndef EMT6RO_COMMON_GRID_H_
#define EMT6RO_COMMON_GRID_H_
#include <cuda_runtime.h>
#include <cstdint>
#include <memory>
#include <type_traits>

namespace emt6ro {

struct Dims {
  int32_t height;
  int32_t width;

  inline __host__ __device__ int32_t vol() const {
    return height * width;
  }

  inline __host__ __device__ bool operator==(Dims rhs) {
    return height == rhs.height && width == rhs.width;
  }

  Dims() = default;

};

struct Coords {
  int32_t r;
  int32_t c;

  __host__ __device__
  uint64_t encode() {
    return *reinterpret_cast<uint64_t*>(this);
  }

  __host__ __device__
  bool operator==(Coords rhs) {
    return r == rhs.r && c == rhs.c;
  }
  
  __host__ __device__
  bool operator!=(Coords rhs) {
    return r != rhs.r || c != rhs.c;
  }

  static __host__ __device__
  Coords decode(uint64_t d) {
    Coords coords;
    coords = *reinterpret_cast<Coords*>(&d);
    return coords;
  }
};

/// Region of interest
struct ROI {
  Coords origin;
  Dims dims;
};

inline ROI bordered(ROI roi, int32_t border = 1) {
  return ROI{{roi.origin.r - border, roi.origin.c - border}, 
             {roi.dims.height + 2*border, roi.dims.width + 2*border}};
}

// class device_iter {
//  public:
  
//   class ICoords {
//     int32_t r; 
//     int32_t c;
//     int32_t i;
//   }

//   class iter {
//     using value_type = ICoords;
//     using pointer  = ICoords*;
//     using reference = ICoords&;
//     using 
//   }
// }

template <typename T>
struct GridView {
  using T_no_cv = typename std::remove_cv<T>::type;

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

  GridView() = default;

  __host__ __device__
  GridView(T *data, Dims dims): data(data), dims(dims) {};

  template <typename T2, 
            typename = std::enable_if_t<std::is_same<T_no_cv, T2>::value>>
  __host__ __device__
  GridView(T2 *data, Dims dims): data(data), dims(dims) {};

  template <typename T2, 
            typename = std::enable_if_t<std::is_same<T_no_cv, T2>::value>>
  __host__ __device__
  GridView(const GridView<T2> &rhs): data(rhs.data), dims(rhs.dims) {}
};

template <typename T>
class HostGrid {
  std::unique_ptr<T[]> data_;
  GridView<T> view_;

 public:
  explicit HostGrid(Dims dims): data_(new T[dims.vol()]), view_{data_.get(), dims} {}

  HostGrid(const HostGrid<T> &rhs)
  : HostGrid(rhs.view_.dims) {
    std::copy(rhs.data_.get(), rhs.data_.get() + rhs.view_.dims.vol(), data_.get());
  }

  GridView<T> view() const {
    return view_;
  }
};

}  // namespace emt6ro

#define GRID_FOR(START_R, START_C, END_R, END_C) \
for (int32_t r = threadIdx.y + START_R; r < END_R; r += blockDim.y) \
  for (int32_t c = threadIdx.x + START_C; c < END_C; c += blockDim.x)

#endif  // EMT6RO_COMMON_GRID_H_
