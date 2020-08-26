#ifndef EMT6RO_COMMON_GRID_H_
#define EMT6RO_COMMON_GRID_H_
#include <cuda_runtime.h>
#include <cstdint>
#include <memory>
#include <type_traits>

namespace emt6ro {

struct Dims {
  int16_t height;
  int16_t width;

  inline __host__ __device__ int32_t vol() const {
    return static_cast<int32_t>(height) * width;
  }

  inline __host__ __device__ bool operator==(Dims rhs) {
    return height == rhs.height && width == rhs.width;
  }

  Dims() = default;

};

struct Coords {
  int16_t r;
  int16_t c;

  __host__ __device__
  uint32_t encode() {
    return (static_cast<uint32_t>(r) << 16) | c;
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
  Coords decode(uint32_t d) {
    Coords coords;
    coords.r = reinterpret_cast<uint16_t*>(&d)[1];
    coords.c = reinterpret_cast<uint16_t*>(&d)[0];
    return coords;
  }
};

struct Coords2 {
  Coords coords[2];

  Coords2() = default;

  __host__ __device__
  Coords2(Coords a, Coords b) {
    coords[0] = a;
    coords[1] = b;
  }

  __host__ __device__
  Coords &operator[](size_t i) {
    return coords[i];
  }

  static __host__ __device__
  Coords2 decode(uint64_t d) {
    Coords2 result;
    result[0] = Coords::decode(reinterpret_cast<uint32_t*>(&d)[1]);
    result[1] = Coords::decode(reinterpret_cast<uint32_t*>(&d)[0]);
    return result;
  }

  __host__ __device__
  uint64_t encode() {
    uint64_t a = static_cast<uint64_t>(coords[0].encode()) << 32;
    uint64_t b = a | static_cast<uint64_t>(coords[1].encode());
    return b;
  }
};

/// Region of interest
struct ROI {
  Coords origin;
  Dims dims;
};

inline ROI bordered(ROI roi, int16_t border = 1) {
  return ROI{{roi.origin.r - border, roi.origin.c - border}, 
             {roi.dims.height + border*2, roi.dims.width + border*2}};
}

template <typename T>
struct GridView {
  using T_no_cv = typename std::remove_cv<T>::type;

  T *data;
  Dims dims;

  inline __host__ __device__  const T& operator()(int16_t r, int16_t c) const {
    return data[r * dims.width + c];
  }

  inline __host__ __device__ T& operator()(int16_t r, int16_t c) {
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

  GridView<T> view() {
    return view_;
  }

  GridView<const T> view() const {
    return GridView<const T>(view_);
  }
};

}  // namespace emt6ro

#define GRID_FOR(START_R, START_C, END_R, END_C) \
for (int16_t r = static_cast<int16_t>(threadIdx.y) + START_R; \
     r < END_R; r += static_cast<int16_t>(blockDim.y)) \
  for (int16_t c = static_cast<int16_t>(threadIdx.x) + START_C; c < END_C; \
       c += static_cast<int16_t>(blockDim.x))

#endif  // EMT6RO_COMMON_GRID_H_
