#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "emt6ro/common/cuda-utils.h"

namespace emt6ro {

__global__ void sumKernel(int n, int *errors) {
  extern __shared__ unsigned sum_aux[];
  auto result = block_reduce(threadIdx.x+1, sum_aux, [](int a, int b) {return a+b;}, n);
  // if (threadIdx.x == 0) *errors = result;
  // return;
  if (result != n * (n + 1) / 2) atomicAdd(errors, 1);
}

TEST(Reduce, AllThreads) {
  int *errors;
  cudaMallocManaged(&errors, sizeof(int));
  *errors = 0;
  sumKernel<<<1, 1024, 32 * sizeof(int), 0>>>(1024, errors);
  cudaDeviceSynchronize();
  ASSERT_EQ(*errors, 0);
}

TEST(Reduce, Partial) {
  int *errors;
  cudaMallocManaged(&errors, sizeof(int));
  *errors = 0;
  sumKernel<<<1, 1024, 32 * sizeof(int), 0>>>(755, errors);
  cudaDeviceSynchronize();
  ASSERT_EQ(*errors, 0);
}

}