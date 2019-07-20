#ifndef SRC_EMT6RO_COMMON_DEVICE_BUFFER_H_
#define SRC_EMT6RO_COMMON_DEVICE_BUFFER_H_

#include <vector>
#include <utility>
#include "emt6ro/common/memory.h"

namespace emt6ro {
namespace device {

template <typename T>
class buffer {
  static_assert(std::is_pod<T>::value, "");
  device::unique_ptr<T> data_;
  size_t size_;

 public:
  buffer(): data_{nullptr}, size_(0) {}

  explicit buffer(size_t count): data_{alloc_unique<T>(count)}, size_(count) {}

  buffer(const T *dev_data, size_t count): data_(alloc_unique<T>(count)), size_(count) {
    cudaMemcpy(data_.get(), dev_data, sizeof(T) * count, cudaMemcpyDeviceToDevice);
  }

  buffer<T> &operator=(const buffer<T> &rhs) {
    size_ = rhs.size();
    data_ = alloc_unique(size_);
    cudaMemcpy(data_.get(), rhs.data(), sizeof(T) * size_, cudaMemcpyDeviceToDevice);
  }

  buffer<T> &operator=(buffer<T> &&rhs) noexcept {
    size_ = rhs.size();
    data_ = std::move(rhs.data_);
  }

  buffer(const buffer<T> &rhs): buffer() {
    *this = rhs;
  }

  buffer(buffer<T> &&rhs) noexcept: buffer() {
    *this = std::move(rhs);
  }

  __host__ __device__ const T *data() const {
    return data_.get();
  }

  __host__ __device__ T *data() {
    return data_.get();
  }

  __device__ const T &operator[](size_t i) const {
    return data_[i];
  }

  __device__ T &operator[](size_t i) {
    return data_[i];
  }

  __host__ __device__ T *begin() {
    return data_.get();
  }

  __host__ __device__ T *end() {
    return data_.get() + size_;
  }

  __host__ __device__ const T *begin() const {
    return data_.get();
  }

  __host__ __device__ const T *end() const {
    return data_.get() + size_;
  }

  __host__ __device__ size_t size() const {
    return size_;
  }

  std::vector<T> toHost() {
    std::vector<T> result(size_);
    cudaMemcpy(result.data(), data_.get(), size_ * sizeof(T), cudaMemcpyDeviceToHost);
    return result;
  }

  static buffer<T> fromHost(const T *data, size_t count) {
    buffer<T> result(count);
    cudaMemcpy(result.data(), data, sizeof(T) * count, cudaMemcpyHostToDevice);
    return result;
  }
};

}  // namespace device
}  // namespace emt6ro

#endif  // SRC_EMT6RO_COMMON_DEVICE_BUFFER_H_
