#ifndef EMT6RO_COMMON_MEMORY_H_
#define EMT6RO_COMMON_MEMORY_H_

#include <cuda_runtime_api.h>
#include <memory>
#include "debug.h"

namespace emt6ro {
namespace device {

struct Stream {
  cudaStream_t stream_;

  Stream() {
    cudaStreamCreate(&stream_);
  }

  Stream(Stream &&rhs): stream_(rhs.stream_) {
    rhs.stream_ = 0;
  }

  Stream& operator=(Stream &&rhs) {
    if (&rhs != this) {
      Release();
      stream_ = rhs.stream_;
      rhs.stream_ = 0;
    }
    return *this;
  }

  ~Stream() {
    Release();
  }

 private:
  void Release() {
    if (stream_) {
      cudaStreamSynchronize(stream_);
      cudaStreamDestroy(stream_);
      KERNEL_DEBUG("stream destroy");
    }
  }
};

class Guard {
  int previous_id{};

 public:
  explicit Guard(int new_id) {
    cudaGetDevice(&previous_id);
    cudaSetDevice(new_id);
  }

  ~Guard() {
    cudaSetDevice(previous_id);
  }
};

struct Deleter {
  int device_id;

  void operator()(void *ptr) {
    Guard device_guard(device_id);
    cudaFree(ptr);
  }
};

template <typename T>
using unique_ptr = std::unique_ptr<T, Deleter>;

template <typename T>
unique_ptr<T> alloc_unique(size_t count = 1) {
  static_assert(std::is_trivially_constructible<T>::value,
      "Allocated type must be default constructable.");
  T *ptr;
  cudaMalloc(&ptr, count * sizeof(T));
  int device_id;
  cudaGetDevice(&device_id);
  return {ptr, Deleter{device_id}};
}

}  // namespace device
}  // namespace emt6ro

#endif  // EMT6RO_COMMON_MEMORY_H_
