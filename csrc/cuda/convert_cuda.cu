// Copyright (c) 2020 Matthias Fey <matthias.fey@tu-dortmund.de>
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

// from https://github.com/rusty1s/pytorch_sparse

#include "convert_cuda.h"
#include "utils.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <c10/util/Optional.h>

#define THREADS 256

template <typename index_t, typename ptr_t>
__global__ void ind2ptr_kernel(const index_t *ind_data, ptr_t *out_data,
                               int64_t M, int64_t numel) {
  ptr_t thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (thread_idx == 0) {
    for (index_t i = 0; i <= ind_data[0]; i++)
      out_data[i] = 0;
  } else if (thread_idx < numel) {
    for (index_t i = ind_data[thread_idx - 1]; i < ind_data[thread_idx]; i++)
      out_data[i + 1] = thread_idx;
  } else if (thread_idx == numel) {
    for (index_t i = ind_data[numel - 1] + 1; i < M + 1; i++)
      out_data[i] = numel;
  }
}

torch::Tensor ind2ptr_cuda(torch::Tensor ind, int64_t M, torch::Tensor out) {
  CHECK_CUDA(ind);
  cudaSetDevice(ind.get_device());
  if (ind.numel() == 0)
    return out.zero_();
  auto blocks = dim3((ind.numel() + 2 + THREADS - 1) / THREADS);
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_INDEX_TYPES(ind.scalar_type(), "ind2ptr_cuda", [&] {
    if (out.dtype() == torch::kInt32) {
      ind2ptr_kernel<<<blocks, THREADS, 0, stream>>>(
          ind.data_ptr<index_t>(), out.data_ptr<int32_t>(), M, ind.numel());
    } else if (out.dtype() == torch::kInt64) {
      ind2ptr_kernel<<<blocks, THREADS, 0, stream>>>(
          ind.data_ptr<index_t>(), out.data_ptr<int64_t>(), M, ind.numel());
    } else {
      out.fill_(NAN);
    }
  });
  return out;
}

static auto registry =
    torch::RegisterOperators().op("lakde::ind2ptr_cuda", &ind2ptr_cuda);
