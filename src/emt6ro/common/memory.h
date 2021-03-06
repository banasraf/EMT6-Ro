#ifndef EMT6RO_COMMON_MEMORY_H_
#define EMT6RO_COMMON_MEMORY_H_

#include <cuda_runtime_api.h>
#include <memory>
#include "emt6ro/common/debug.h"

namespace emt6ro {
namespace device {

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


struct Stream {
  cudaStream_t stream_;
  int device_id_;

  Stream() {
    cudaGetDevice(&device_id_);
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
      Guard dg(device_id_);
      cudaStreamSynchronize(stream_);
      cudaStreamDestroy(stream_);
      KERNEL_DEBUG("stream destroy");
    }
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
      "Allocated type must be trivially constructable.");
  T *ptr;
  cudaMalloc(&ptr, count * sizeof(T));
  int device_id;
  cudaGetDevice(&device_id);
  return {ptr, Deleter{device_id}};
}

}  // namespace device
}  // namespace emt6ro

#endif  // EMT6RO_COMMON_MEMORY_H_
