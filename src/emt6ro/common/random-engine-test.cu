#include <gtest/gtest.h>
#include "emt6ro/common/random-engine.h"

namespace emt6ro {

__global__ void fillUniform(float *data, curandState_t *states) {
  auto x = blockIdx.x * blockDim.x + threadIdx.x;
  CuRandEngine engine(&states[x]);
  data[x] = engine.uniform();
}

TEST(RandomGenerator, FillUniform) {
  device::buffer<float> data(2 * 1024);
  std::vector<uint32_t> h_seeds(2*1024);
  for (size_t i = 0; i < 2 * 1024; ++i) h_seeds[i] = i;
  auto d_seeds = device::buffer<uint32_t>::fromHost(h_seeds.data(), 2*1024);
  CuRandEngineState state(2 * 1024, d_seeds.data());
  fillUniform<<<2, 1024>>>(data.data(), state.states());
  auto h_data = data.toHost();
  for (size_t i = 0; i < 2 * 1024; ++i) {
    ASSERT_TRUE(h_data[i] >= 0.0 && h_data[i] <= 1.0);
  }
}

}  // namespace emt6ro