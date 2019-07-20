#ifndef SRC_EMT6RO_COMMON_MEMORY_H_
#define SRC_EMT6RO_COMMON_MEMORY_H_

#include <cuda_runtime_api.h>
#include <memory>

namespace emt6ro {
namespace device {

struct Deleter {
  void operator()(void *ptr) {
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
  return {ptr, Deleter{}};
}

}  // namespace device
}  // namespace emt6ro

#endif  // SRC_EMT6RO_COMMON_MEMORY_H_
