#ifndef EMT6RO_COMMON_CUDA_UTILS_H_
#define EMT6RO_COMMON_CUDA_UTILS_H_

#include <cuda_runtime.h>

#define FULL_MASK 0xffffffff

namespace emt6ro {

static constexpr uint32_t CuBlockDimX = 16;
static constexpr uint32_t CuBlockDimY = 16;
static constexpr uint32_t SitesPerThread = 16;

template <typename L,
          typename R,
          bool enable = std::is_integral<L>::value && std::is_integral<R>::value,
          typename = std::enable_if_t<enable>>
__device__ __host__
L div_ceil(L lhs, R rhs) {
  return (lhs + rhs - 1) / rhs;
}

/// The first thread in a warp returns a reduced value
template <typename T, typename Op>
__device__ T warp_reduce(T val, const Op &op, int i, int n) {
  #pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    unsigned mask = __ballot_sync(FULL_MASK, i + offset < n);
    val = op(val, __shfl_down_sync(mask, val, offset));
  }
  return val;
}

template <typename T, typename Op>
__device__ T block_reduce(T val, T *shared_mem, const Op &op, int n = -1) {
  if (n < 0) n = blockDim.x * blockDim.y * blockDim.z;
  auto tid = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
  val = warp_reduce(val, op, tid, n);
  if (tid < n && tid % 32 == 0) shared_mem[tid / 32] = val;
  __syncthreads();
  for (int size = div_ceil(n, 32); size > 1; size = div_ceil(size, 32)) {
    bool thread_active = tid < size;
    if (thread_active) val = shared_mem[tid];
    val = warp_reduce(val, op, tid, n);
    if (thread_active && tid % 32 == 0) shared_mem[tid / 32] = val;
    __syncthreads();
  }
  return shared_mem[0];
}

__device__
inline int blockVol() {
  return blockDim.x * blockDim.y * blockDim.z;
}

__device__
inline int threadId() {
  return (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
}

}  // namespace emt6ro
#endif  // EMT6RO_COMMON_CUDA_UTILS_H_
