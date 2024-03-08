#include "chol_quad_form.h"
#include "utils.cuh"
#include <ATen/cuda/CUDAContext.h>

template <typename scalar_t>
__global__ void chol_quad_form_kernel(
    const scalar_t *W_triu_data, const scalar_t *Xn_data,
    const scalar_t *Xm_data, scalar_t *out_data, const int W_triu_stride1,
    const int W_triu_stride2, const int W_triu_stride3, const int Xn_stride1,
    const int Xn_stride2, const int Xm_stride1, const int Xm_stride2,
    const int out_stride1, const int out_stride2, const int N, const int M,
    const int D) {
  // 256 = 16 * 16 threads
  // 8 warps = 16 half-warps
  __shared__ scalar_t W_triu_blocks[16][16][16]; // [n, j, i]
  __shared__ scalar_t Xn_block[16][16];          // [n, j]
  __shared__ scalar_t
      Xm_block[16][16 + 1]; // [n, j] (+1 to avoid bank conflicts)

  int m = blockDim.x * blockIdx.x + threadIdx.x;
  int n = blockDim.y * blockIdx.y + threadIdx.y;
  int mrow = blockDim.x * blockIdx.x + threadIdx.y;
  int nrow = blockDim.y * blockIdx.y + threadIdx.y;

  scalar_t result = 0;
  scalar_t inners[16];

  for (int ci = 0; ci < D; ci += 16) {
#pragma unroll
    for (int wi = 0; wi < 16; wi++)
      inners[wi] = 0;

    for (int cj = 0; cj <= ci; cj += 16) { // triangular => cj <= ci
      if (nrow < N && cj + threadIdx.x < D)
        Xn_block[threadIdx.y][threadIdx.x] = __ldg(
            Xn_data + nrow * Xn_stride1 + (cj + threadIdx.x) * Xn_stride2);
      else
        Xn_block[threadIdx.y][threadIdx.x] = 0;

      if (mrow < M && cj + threadIdx.x < D)
        Xm_block[threadIdx.y][threadIdx.x] = __ldg(
            Xm_data + mrow * Xm_stride1 + (cj + threadIdx.x) * Xm_stride2);
      else
        Xm_block[threadIdx.y][threadIdx.x] = 0;
#pragma unroll
      for (int wj = 0; wj < 16; wj++)
        if (nrow < N && cj + wj < D && ci + threadIdx.x < D)
          W_triu_blocks[wj][threadIdx.y][threadIdx.x] = __ldg(
              W_triu_data + nrow * W_triu_stride1 + (cj + wj) * W_triu_stride2 +
              (ci + threadIdx.x) * W_triu_stride3);
        else
          W_triu_blocks[wj][threadIdx.y][threadIdx.x] = 0;

      __syncthreads();
#pragma unroll
      for (int wj = 0; wj < 16; wj++) {
        scalar_t Xnj = Xn_block[threadIdx.y][wj];
        scalar_t Xmj = Xm_block[threadIdx.x][wj];
#pragma unroll
        for (int wi = 0; wi < 16; wi++) {
          scalar_t W_triu_nji = W_triu_blocks[wj][threadIdx.y][wi];
          inners[wi] += W_triu_nji * (Xmj - Xnj);
        }
      }
      __syncthreads();
    }

#pragma unroll
    for (int wi = 0; wi < 16; wi++)
      result += inners[wi] * inners[wi];
  }

  if (n < N && m < M) {
    *(out_data + n * out_stride1 + m * out_stride2) = result;
  }
}

torch::Tensor chol_quad_form_cuda(torch::Tensor W_triu, torch::Tensor Xn,
                                  torch::Tensor Xm, torch::Tensor out) {
  CHECK_CUDA(W_triu);
  CHECK_CUDA(Xn);
  CHECK_CUDA(Xm);

  CHECK_INPUT(W_triu.dim() == 3);
  CHECK_INPUT(Xn.dim() == 2);
  CHECK_INPUT(Xm.dim() == 2);

  auto N = Xn.size(0);
  auto M = Xm.size(0);
  auto D = Xn.size(1);

  CHECK_INPUT(Xm.size(1) == D);
  CHECK_INPUT(W_triu.size(0) == N);
  CHECK_INPUT(W_triu.size(1) == D);
  CHECK_INPUT(W_triu.size(2) == D);

  cudaSetDevice(W_triu.get_device());

  auto out_stride1 = out.stride(0);
  auto out_stride2 = out.stride(1);

  auto blocks = dim3((M + 15) / 16, (N + 15) / 16);
  auto threads = dim3(16, 16);
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND(
      torch::kHalf, W_triu.scalar_type(), "chol_quad_form_cuda", [&] {
        chol_quad_form_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            W_triu.data_ptr<scalar_t>(), Xn.data_ptr<scalar_t>(),
            Xm.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), W_triu.stride(0),
            W_triu.stride(1), W_triu.stride(2), Xn.stride(0), Xn.stride(1),
            Xm.stride(0), Xm.stride(1), out_stride1, out_stride2, N, M, D);
      });
  return out;
}

static auto registry = torch::RegisterOperators().op(
    "lakde::chol_quad_form_cuda", &chol_quad_form_cuda);
