import os
import os.path as osp
import sys

import torch
import torch.utils.cpp_extension
from torch.__config__ import parallel_info


# slow fallback kernels
def ind2ptr(row, M, dtype=None, out=None):
    if out is None:
        out = row.new_zeros(M + 1, dtype=dtype)
    inds, counts = torch.unique_consecutive(row, return_counts=True)
    inds += 1
    out[inds.long()] = counts.to(inds.dtype)
    return torch.cumsum(out, dim=0, out=out)


def chol_quad_form(W_triu, Xn, Xm):
    diff = Xm[None, :] - Xn[:, None]
    LX = W_triu.mT @ diff.mT
    return LX.pow_(2).sum(dim=-2)


def calc_W_inv(rowptr, col, value, Xn, Xm):
    N, D = Xn.shape
    out = value.new_empty(N, D, D)
    row_start = rowptr[0]
    for i in range(N):
        row_end = rowptr[i + 1]
        c = col[row_start:row_end].long()
        v = value[row_start:row_end]
        xn = Xn[i]
        diff = Xm[c] - xn
        rnm_diff = v[:, None] * diff
        out[i] = rnm_diff.mT @ diff
        row_start = row_end
    return out


if not "LAKDE_NO_EXT" in os.environ or os.environ["LAKDE_NO_EXT"] == "0":
    ext_dir = osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), "csrc")

    cuda_sources = [
        "cuda/W_inv_cuda.cu",
        "cuda/convert_cuda.cu",
        "cuda/chol_quad_form.cu",
    ]

    sources = cuda_sources
    sources = [osp.join(ext_dir, s) for s in sources]

    extra_cflags = ["-O2"]
    extra_cuda_cflags = [
        "-arch=sm_60",
        "--expt-relaxed-constexpr",
        "-O2",
        "-gencode=arch=compute_60,code=sm_60",
        "-gencode=arch=compute_61,code=sm_61",
        "-gencode=arch=compute_70,code=sm_70",
        "-gencode=arch=compute_75,code=sm_75",
        "-gencode=arch=compute_80,code=sm_80",
        "-gencode=arch=compute_86,code=sm_86",
    ]
    if "CXX" in os.environ:
        extra_cuda_cflags.append("-ccbin=" + os.environ["CXX"])

    info = parallel_info()
    if "backend: OpenMP" in info and "OpenMP not found" not in info:
        extra_cflags += ["-DAT_PARALLEL_OPENMP"]
        if sys.platform == "win32":
            extra_cflags += ["/openmp"]
        else:
            extra_cflags += ["-fopenmp"]
    else:
        print("Compiling without OpenMP...")

    torch.utils.cpp_extension.load(
        name="lakde",
        sources=sources,
        verbose=True,
        with_cuda=True,
        extra_cflags=extra_cflags,
        extra_cuda_cflags=extra_cuda_cflags,
        is_python_module=False,
    )

    ind2ptr_torch = ind2ptr
    calc_W_inv_torch = calc_W_inv
    chol_quad_form_torch = chol_quad_form

    def ind2ptr(ind, M, out=None):
        assert ind.is_contiguous()
        if out is None:
            out = ind.new_empty(M + 1)
        if ind.is_cuda:
            return torch.ops.lakde.ind2ptr_cuda(ind, M, out)
        else:
            return ind2ptr_torch(ind, M, out=out)

    def calc_W_inv(rowptr, col, value, Xn, Xm):
        assert rowptr.is_contiguous()
        assert col.is_contiguous()
        assert value.is_contiguous()
        assert Xn.is_contiguous()
        assert Xm.is_contiguous()
        if rowptr.is_cuda:
            N, D = Xn.shape
            out = value.new_empty(N, D, D)
            return torch.ops.lakde.W_inv_cuda(rowptr, col, value, Xn, Xm, out)
        else:
            return calc_W_inv_torch(rowptr, col, value, Xn, Xm)

    def chol_quad_form(W_triu, Xn, Xm):
        if W_triu.is_cuda:
            out = W_triu.new_empty(Xn.size(0), Xm.size(0))
            return torch.ops.lakde.chol_quad_form_cuda(W_triu, Xn, Xm, out)
        else:
            return chol_quad_form_torch(W_triu, Xn, Xm)
else:
    print("Custom CUDA kernels not enabled, using slower fallback kernels...")
