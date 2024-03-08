#include "W_inv_cuda.h"
#include "utils.cuh"
#include <ATen/cuda/CUDAContext.h>

#define THREADS 256

template <typename scalar_t, typename rowptr_t, typename index_t>
__global__ void W_inv_kernel(const rowptr_t *rowptr_data,
                             const index_t *col_data,
                             const scalar_t *value_data,
                             const scalar_t *Xn_data, const scalar_t *Xm_data,
                             scalar_t *out_data, const int N, const int D) {

  // 1 warp (32 threads) per row of rnm
  int thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
  int row = thread_idx >> 5;                     // thread_idx / 32
  int lane_idx = thread_idx & (32 - 1);          // thread_idx % 32
  int linear_idx = (blockIdx.y << 5) + lane_idx; // blockIdx.y * 32 + lane_idx

  // linear to triangular indices
  int j = ((int)sqrt((double)(8 * linear_idx + 1)) - 1) / 2;
  int i = linear_idx - j * (j + 1) / 2;

  if (row < N) {
    rowptr_t row_start = __ldg(rowptr_data + row);
    rowptr_t row_end = __ldg(rowptr_data + row + 1);

    scalar_t Xni = __ldg(Xn_data + row * D + i);
    scalar_t Xnj = __ldg(Xn_data + row * D + j);

    int col, cols[32];
    scalar_t rnm, rnms[32];
    scalar_t result = 0;

    for (rowptr_t c = row_start; c < row_end; c += 32) {
      if (c + lane_idx < row_end) {
        // coalesced access into col and value
        col = __ldg(col_data + c + lane_idx);
        rnm = __ldg(value_data + c + lane_idx);
      } else {
        col = -1;
        rnm = 0;
      }

#pragma unroll
      for (int w = 0; w < 32; w++) {
        // exchange all 32 col and rnms in warp
        cols[w] = __shfl_sync_patched(FULL_MASK, col, w);
        rnms[w] = __shfl_sync_patched(FULL_MASK, rnm, w);
      }

#pragma unroll
      for (int w = 0; w < 32; w++) {
        if (linear_idx < D * (D + 1) / 2 && cols[w] != -1) {
          scalar_t Xmi = __ldg(Xm_data + cols[w] * D + i);
          scalar_t Xmj = __ldg(Xm_data + cols[w] * D + j);
          result += rnms[w] * (Xmi - Xni) * (Xmj - Xnj);
        }
      }
    }

    if (linear_idx < D * (D + 1) / 2) {
      // symmetric write to out
      *(out_data + row * D * D + i * D + j) = result;
      *(out_data + row * D * D + j * D + i) = result;
    }
  }
}

torch::Tensor W_inv_cuda(torch::Tensor rowptr, torch::Tensor col,
                         torch::Tensor value, torch::Tensor Xn,
                         torch::Tensor Xm, torch::Tensor out) {
  CHECK_CUDA(rowptr);
  CHECK_CUDA(col);
  CHECK_CUDA(value);
  CHECK_CUDA(Xn);
  CHECK_CUDA(Xm);

  CHECK_INPUT(rowptr.dim() == 1);
  CHECK_INPUT(col.dim() == 1);
  CHECK_INPUT(value.size(0) == col.size(0));
  CHECK_INPUT(value.dim() == 1);

  CHECK_INPUT(Xn.dim() == 2);
  CHECK_INPUT(Xm.dim() == 2);

  auto N = Xn.size(0);
  auto D = Xn.size(1);

  CHECK_INPUT(rowptr.numel() - 1 == N);
  CHECK_INPUT(Xm.size(1) == D);

  CHECK_INPUT(Xn.is_contiguous());
  CHECK_INPUT(Xm.is_contiguous());
  CHECK_INPUT(out.is_contiguous());

  cudaSetDevice(rowptr.get_device());

  auto blocks =
      dim3((32 * N + THREADS - 1) / THREADS, (D * (D + 1) / 2 + 31) / 32);
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND(
      torch::kHalf, value.scalar_type(), "W_inv_cuda", [&] {
        if (rowptr.scalar_type() == torch::kInt32 &&
            col.scalar_type() == torch::kInt32) {
          W_inv_kernel<scalar_t, int32_t, int32_t>
              <<<blocks, THREADS, 0, stream>>>(
                  rowptr.data_ptr<int32_t>(), col.data_ptr<int32_t>(),
                  value.data_ptr<scalar_t>(), Xn.data_ptr<scalar_t>(),
                  Xm.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), N, D);
        } else if (rowptr.scalar_type() == torch::kInt32 &&
                   col.scalar_type() == torch::kInt64) {
          W_inv_kernel<scalar_t, int32_t, int64_t>
              <<<blocks, THREADS, 0, stream>>>(
                  rowptr.data_ptr<int32_t>(), col.data_ptr<int64_t>(),
                  value.data_ptr<scalar_t>(), Xn.data_ptr<scalar_t>(),
                  Xm.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), N, D);
        } else if (rowptr.scalar_type() == torch::kInt64 &&
                   col.scalar_type() == torch::kInt32) {
          W_inv_kernel<scalar_t, int64_t, int32_t>
              <<<blocks, THREADS, 0, stream>>>(
                  rowptr.data_ptr<int64_t>(), col.data_ptr<int32_t>(),
                  value.data_ptr<scalar_t>(), Xn.data_ptr<scalar_t>(),
                  Xm.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), N, D);
        } else if (rowptr.scalar_type() == torch::kInt64 &&
                   col.scalar_type() == torch::kInt64) {
          W_inv_kernel<scalar_t, int64_t, int64_t>
              <<<blocks, THREADS, 0, stream>>>(
                  rowptr.data_ptr<int64_t>(), col.data_ptr<int64_t>(),
                  value.data_ptr<scalar_t>(), Xn.data_ptr<scalar_t>(),
                  Xm.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), N, D);
        } else {
          out.fill_(NAN);
        }
      });
  return out;
}

static auto registry =
    torch::RegisterOperators().op("lakde::W_inv_cuda", &W_inv_cuda);
