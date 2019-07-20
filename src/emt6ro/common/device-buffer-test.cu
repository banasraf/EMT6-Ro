#include <gtest/gtest.h>
#include "emt6ro/common/device-buffer.h"

namespace emt6ro {

namespace {
const size_t BLOCK_SIZE = 128;

__global__ void fill(int *data) {
  data[threadIdx.x] = threadIdx.x;
}

__global__ void isSequence(const int *data, size_t size, bool *result) {
  *result = true;
  for (int i = 0; i < size; ++i) {
    if (data[i] != i) {
      *result = false;
      break;
    }
  }
}
}  // namespace

TEST(DeviceBuffer, ToHost) {
  device::buffer<int> buf(BLOCK_SIZE);
  fill<<<1, BLOCK_SIZE>>>(buf.data());
  auto host_data = buf.toHost();
  for (int i = 0; i < BLOCK_SIZE; ++i) {
    ASSERT_EQ(host_data[i], i);
  }
}

TEST(DeviceBuffer, FromHost) {
  std::vector<int> seq(BLOCK_SIZE);
  for (int i = 0; i < BLOCK_SIZE; ++i) {
    seq[i] = i;
  }
  auto dev = device::buffer<int>::fromHost(seq.data(), seq.size());
  auto result = device::buffer<bool>(1);
  isSequence<<<1, 1>>>(dev.data(), BLOCK_SIZE, result.data());
  ASSERT_TRUE(result.toHost()[0]);
}

TEST(DeviceBuffer, Copy) {
  device::buffer<int> buf(BLOCK_SIZE);
  fill<<<1, BLOCK_SIZE>>>(buf.data());
  device::buffer<int> buf2 =  buf;
  auto host_data = buf2.toHost();
  for (int i = 0; i < BLOCK_SIZE; ++i) {
    ASSERT_EQ(host_data[i], i);
  }
}

TEST(DeviceBuffer, Move) {
  device::buffer<int>  buf(BLOCK_SIZE);
  fill<<<1, BLOCK_SIZE>>>(buf.data());
  device::buffer<int> buf2 = std::move(buf);
  auto host_data = buf2.toHost();
  for (int i = 0; i < BLOCK_SIZE; ++i) {
    ASSERT_EQ(host_data[i], i);
  }
}

}  // namespace emt6ro
