#ifndef EMT6RO_COMMON_DEBUG_H_
#define EMT6RO_COMMON_DEBUG_H_

#include <iostream>
#include <string>

#ifdef NDEBUG
constexpr bool _DEBUG = false;
#else
constexpr bool _DEBUG = true;
#endif

#define KERNEL_DEBUG(name) \
if (_DEBUG) { \
  cudaDeviceSynchronize(); \
  auto code = cudaPeekAtLastError();\
  if (code != cudaSuccess) { \
    std::string err(cudaGetErrorString(code)); \
    throw std::runtime_error("CUDA error: " + err + " in " name); \
  } \
}

#endif  // EMT6RO_COMMON_DEBUG_H_
