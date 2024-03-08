#pragma once

#include <torch/extension.h>

#define FULL_MASK 0xffffffff

template <typename scalar_t>
__forceinline__ __device__ scalar_t __shfl_sync_patched(const unsigned int mask,
                                                        const scalar_t var,
                                                        const int width) {
  return __shfl_sync(mask, var, width);
}

template <>
__forceinline__ __device__ at::Half __shfl_sync_patched(const unsigned int mask,
                                                        const at::Half var,
                                                        const int width) {
  // resolve ambiguous implicit conversion with half type
  return __shfl_sync(mask, var.operator __half(), width);
}

#define CHECK_CUDA(x)                                                          \
  AT_ASSERTM(x.device().is_cuda(), #x " must be CUDA tensor")
#define CHECK_INPUT(x) AT_ASSERTM(x, "Input mismatch")
