#include "emt6ro/statistics/statistics.h"
#include <cuda_runtime.h>
#include "emt6ro/common/debug.h"

namespace emt6ro {

namespace detail {

__global__ void countLivingKernel(uint32_t *results, Site *data, uint32_t vol) {
  extern __shared__ uint32_t parts[];
  parts[threadIdx.x] = 0;
  auto tumor = data + blockIdx.x * vol;
  for (int i = threadIdx.x; i < vol; i += blockDim.x) {
    parts[threadIdx.x] += static_cast<uint32_t>(tumor[i].isOccupied());
  }
  __syncthreads();
  int ti = threadIdx.x;
  for (int d = 1; d < blockDim.x; d *= 2) {
    if (ti % 2 == 0) {
      parts[threadIdx.x] += parts[threadIdx.x + d];
      ti /= 2;
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) results[blockIdx.x] = parts[0];
}

}

void countLiving(uint32_t *results, Site *data, Dims dims, uint32_t batch_size) {
  auto vol = dims.vol();
  auto block_size = (vol < 1024) ? vol : 1024;
  detail::countLivingKernel
    <<<batch_size, block_size, block_size * sizeof(uint32_t)>>>
    (results, data, vol);
  KERNEL_DEBUG("count living")
}

}
