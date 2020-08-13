#ifndef EMT6RO_COMMON_DEVICE_BUFFER_H_
#define EMT6RO_COMMON_DEVICE_BUFFER_H_

#include <vector>
#include <utility>
#include <memory>
#include "emt6ro/common/memory.h"

namespace emt6ro {
namespace device {

template <typename T>
class buffer {
  static_assert(std::is_pod<T>::value, "device::buffer can hold only POD types");
  device::unique_ptr<T> data_;
  size_t size_;

 public:
  buffer(): data_{nullptr}, size_(0) {}

  explicit buffer(size_t count): data_{alloc_unique<T>(count)}, size_(count) {}

  /**
   * Allocate a new buffer and copy device data to it.
   * The copy is done asynchronously if the stream is provided.
   * @param dev_data - pointer to device data
   * @param count - number of elements to allocate
   * @param stream - cuda stream which the copy should be scheduled to
   */
  buffer(const T *dev_data, size_t count, cudaStream_t stream = nullptr)
  : data_(alloc_unique<T>(count))
  , size_(count) {
    cudaMemcpyAsync(data_.get(), dev_data, sizeof(T) * count, cudaMemcpyDeviceToDevice, stream);
    if (!stream)
      cudaStreamSynchronize(stream);
  }

  /**
   * Allocate a new buffer and synchronously copy data from the `rhs` buffer.
   * @param rhs
   */
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

  /**
   * Allocate a new host buffer and copy the data into it.
   * The copy is done asynchronously if the stream is provided.
   * @param stream - cuda stream which the copy should be scheduled to
   * @return allocated host buffer
   */
  std::unique_ptr<T[]> toHost(cudaStream_t stream = nullptr) {
    std::unique_ptr<T[]> result(new T[size_]);
    cudaMemcpyAsync(result.get(), data_.get(), size_ * sizeof(T), cudaMemcpyDeviceToHost, stream);
    if (!stream)
      cudaStreamSynchronize(stream);
    return result;
  }

  /**
   * Allocate a new buffer and copy the host data into it.
   * The copy is done asynchronously if the stream is provided.
   * @param data - host data to be copied
   * @param count - number of elements
   * @param stream - cuda stream which the copy should be scheduled to
   * @return new device buffer
   */
  static buffer<T> fromHost(const T *data, size_t count, cudaStream_t stream = nullptr) {
    buffer<T> result(count);
    cudaMemcpyAsync(result.data(), data, sizeof(T) * count, cudaMemcpyHostToDevice, stream);
    if (!stream)
      cudaStreamSynchronize(stream);
    return result;
  }

  /**
   * Copy host data into the buffer.
   * The copy is done asynchronously if the stream is provided.
   * @param data - host source data
   * @param count - number of elements to copy
   * @param dest_offset - copy destination offset
   * @param stream - cuda stream which the copy should be scheduled to
   */
  void copyHost(const T *data, size_t count,
                size_t dest_offset = 0, cudaStream_t stream = nullptr) {
    cudaMemcpyAsync(data_.get() + dest_offset, data, sizeof(T) * count,
                    cudaMemcpyHostToDevice, stream);
    if (!stream)
      cudaStreamSynchronize(stream);
  }

  void copyToHost(T *data, size_t count = -1,
                  size_t source_offset = 0, cudaStream_t stream = nullptr) {
    if (count == -1) count = size_;
    cudaMemcpyAsync(data, data_.get() + source_offset, sizeof(T) * count,
                    cudaMemcpyDeviceToHost, stream);
    if (!stream)
      cudaStreamSynchronize(stream);
  }
};

}  // namespace device
}  // namespace emt6ro

#endif  // EMT6RO_COMMON_DEVICE_BUFFER_H_
