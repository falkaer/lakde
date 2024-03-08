#pragma once

#include <torch/extension.h>

torch::Tensor W_inv_cuda(torch::Tensor rowptr, torch::Tensor col,
                         torch::Tensor value, torch::Tensor Xn,
                         torch::Tensor Xm, torch::Tensor out);
