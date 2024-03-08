#pragma once

#include <torch/extension.h>

torch::Tensor chol_quad_form_cuda(torch::Tensor W_triu, torch::Tensor Xn,
                                  torch::Tensor Xm, torch::Tensor out);
