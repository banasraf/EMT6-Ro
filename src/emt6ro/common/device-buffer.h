#ifndef SRC_EMT6RO_COMMON_DEVICE_BUFFER_H_
#define SRC_EMT6RO_COMMON_DEVICE_BUFFER_H_

#include <vector>
#include <utility>
#include <memory>
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
    data_ = alloc_unique<T>(size_);
    cudaMemcpy(data_.get(), rhs.data(), sizeof(T) * size_, cudaMemcpyDeviceToDevice);
    return *this;
  }

  buffer<T> &operator=(buffer<T> &&rhs) noexcept {
    size_ = rhs.size();
    data_ = std::move(rhs.data_);
    return *this;
  }

  buffer(const buffer<T> &rhs): buffer() {
    *this = rhs;
  }

  buffer(buffer<T> &&rhs) noexcept: buffer() {
    *this = std::move(rhs);
  }

  const T *data() const {
    return data_.get();
  }

  T *data() {
    return data_.get();
  }

  size_t size() const {
    return size_;
  }

  std::unique_ptr<T[]> toHost() {
    std::unique_ptr<T[]> result(new T[size_]);
    cudaMemcpy(result.get(), data_.get(), size_ * sizeof(T), cudaMemcpyDeviceToHost);
    return result;
  }

  static buffer<T> fromHost(const T *data, size_t count) {
    buffer<T> result(count);
    cudaMemcpy(result.data(), data, sizeof(T) * count, cudaMemcpyHostToDevice);
    return result;
  }

  void copyHost(const T *data, size_t count, size_t dest_offset = 0) {
    cudaMemcpy(data_.get() + dest_offset, data, sizeof(T) * count, cudaMemcpyHostToDevice);
  }
};

}  // namespace device
}  // namespace emt6ro

#endif  // SRC_EMT6RO_COMMON_DEVICE_BUFFER_H_
