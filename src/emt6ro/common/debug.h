#ifndef EMT6RO_COMMON_DEBUG_H_
#define EMT6RO_COMMON_DEBUG_H_

#include <iostream>

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
    std::cout << "error in " << name << ": " << cudaGetErrorString(code) << std::endl; \
    exit(-1); \
  } \
}

#endif  // EMT6RO_COMMON_DEBUG_H_
