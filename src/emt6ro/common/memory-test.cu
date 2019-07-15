#include <gtest/gtest.h>
#include <vector>
#include "emt6ro/common/memory.h"

namespace emt6ro {

const size_t SIZE = 128;

__global__ void scale(float *data, float scalar) {
  data[threadIdx.x] *= scalar;
}

TEST(MemoryAllocation, SimpleAllocation) {
  std::vector<float> host_data(SIZE, 0.5);
  auto ptr = device::alloc_unique<float>(SIZE);
  cudaMemcpy(ptr.get(), host_data.data(), SIZE * sizeof(float), cudaMemcpyHostToDevice);
  scale<<<1, SIZE>>>(ptr.get(), 2.);
  std::vector<float> host_result(SIZE);
  cudaMemcpy(host_result.data(), ptr.get(), SIZE * sizeof(float), cudaMemcpyDeviceToHost);
  for (auto elem: host_result) {
    ASSERT_FLOAT_EQ(elem, 1.);
  }
}

}  // namespace emt6ro
