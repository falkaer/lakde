import math

import torch


# ensure lgamma always computed with 64 bits,
# 32-bit version becomes inaccurate for large a
def lgamma(a):
    return torch.lgamma(a.double()).to(a.dtype)


# multivariate log gamma function
def mvlgamma(a, D):
    i = torch.linspace(1, D, D, dtype=a.dtype, device=a.device)
    i = i.expand(a.shape + i.shape)
    a = a.unsqueeze(-1)
    coeff = D * (D - 1) / 4 * math.log(math.pi)
    return coeff + torch.sum(lgamma(a + (1 - i) / 2), dim=-1)


# multivariate digamma function
def mvdigamma(a, D):
    i = torch.linspace(1, D, D, dtype=a.dtype, device=a.device)
    i = i.expand(a.shape + i.shape)
    a = a.unsqueeze(-1)
    return torch.sum(torch.digamma(a + (1 - i) / 2), dim=-1)


# compute log(det(L @ L.T)) = 2 * sum(log(diag(L)))
def chol_logdet(L):
    return 2 * L.diagonal(dim1=-2, dim2=-1).log().sum(dim=-1)


# tr((L @ L.T) @ (R @ R.T)) = sum((L.T @ R) * (L.T @ R))
def chol_mm_trace(L, R):
    LR = L.mT @ R
    return LR.pow_(2).sum(dim=(-1, -2))


# log(exp(a) - exp(b)) = a + log(1 - exp(b - a))
def logsubexp(a, b):  # threshold so a <= b => -inf
    return a + torch.log1p(torch.clamp_min(-torch.exp(b - a), -1))


def xlogx(x, out=None):
    return torch.xlogy(x, x, out=out)


def tril_inverse(L):
    D = L.size(-1)
    I = torch.eye(D, out=L.new_empty(D, D)).expand_as(L).contiguous()
    return torch.linalg.solve_triangular(L, I, out=I, upper=False)


def tril_to_vec(L, out=None):
    D = L.size(-1)
    if out is None:
        Dv = D * (D + 1) // 2
        out = L.new_empty(L.shape[:-2] + (Dv,))
    inds = torch.tril_indices(D, D, device=out.device)
    out[...] = L[..., inds[0], inds[1]]
    return out


def vec_to_tril(Lv, out=None, symmetric=False):
    Dv = Lv.size(-1)
    D = (math.isqrt(8 * Dv + 1) - 1) // 2
    if out is None:
        out = Lv.new_zeros(Lv.shape[:-1] + (D, D))
    inds = torch.tril_indices(D, D, device=out.device)
    out[..., inds[0], inds[1]] = Lv
    if symmetric:
        out[..., inds[1], inds[0]] = Lv
    return out


def vec_to_diag(Lv, out=None):
    Dv = Lv.size(-1)
    D = (math.isqrt(8 * Dv + 1) - 1) // 2
    if out is None:
        out = Lv.new_empty(Lv.shape[:-2] + (D,))
    inds = torch.arange(D, device=out.device)
    inds += torch.cumsum(inds, dim=0)
    out[...] = Lv[..., inds]
    return out


# currently if cholesky is invoked on a batch of 1 matrix pytorch will
# use the single-matrix algorithm, which produces different results
# so we have to pad the matrix and force the batched algorithm
def _check_decomposable_(X, info):
    single = X.size(0) == 1
    if single:  # pad the batch to 2
        shape = list(X.shape)
        shape[0] = 2
        X = X.resize_(shape)
        X[1] = X[0]
        info.resize_((2,))

    info = torch.linalg.cholesky_ex(X, out=(X, info))[1]
    if single:  # remove padding
        shape[0] = 1
        X.resize_(shape)
        info = info.resize_((1,))
    return info != 0


# ref: https://stackoverflow.com/questions/43238173/python-convert-matrix-to-positive-semi-definite
def stabilize_(X, out=None):
    if out is None:
        out = torch.empty_like(X)
    spacing = torch.finfo(X.dtype).eps
    jitter = X.new_zeros(X.shape[:-2])
    info = X.new_empty(X.shape[:-2], dtype=torch.int32)
    mask = X.new_ones(X.shape[:-2], dtype=torch.bool)
    for k in range(0, 32):
        # mineig = torch.linalg.eigvals(Xmasked).abs().min(dim=-1).values
        mineig = spacing
        jitter[mask] += mineig * 2**k
        out.copy_(X)
        out.diagonal(dim1=-2, dim2=-1).add_(jitter[..., None])
        # check if decomposable
        mask = _check_decomposable_(out, info)
        if not torch.any(mask):
            return jitter
    raise RuntimeError("Stabilization did not converge")


def stabilized_cholesky(X, jitter=None, out=None, verbose=False):
    info = X.new_empty(X.shape[:-2], dtype=torch.int32)
    if out is None:
        out = torch.empty_like(X)

    if jitter is not None and torch.count_nonzero(jitter) != 0:
        out.copy_(X)
        out.diagonal(dim1=-2, dim2=-1).add_(jitter[..., None])
        L, info = torch.linalg.cholesky_ex(out, out=(out, info), check_errors=False)
    else:
        L, info = torch.linalg.cholesky_ex(X, out=(out, info), check_errors=False)

    if not torch.any(info):
        return L, jitter

    if jitter is None:
        jitter = X.new_zeros(X.shape[:-2])
    else:
        jitter = jitter.clone()

    mask = info != 0
    if verbose:
        if X.ndim == 2:
            print("Failed to decompose matrix with jitter {}".format(jitter))
        else:
            inds = torch.arange(X.size(0), device=mask.device)[mask]
            print(
                "Failed to decompose matrices at indices {} with jitter {}".format(
                    inds, jitter[mask]
                )
            )

    Xmasked = X[mask]
    jitter[mask] = stabilize_(Xmasked, out.resize_as_(Xmasked))
    out.resize_as_(X)
    if verbose:
        if X.ndim == 2:
            print("Found new jitter value {}".format(jitter))
        else:
            print("Found new jitter values {}".format(jitter[mask]))

    return stabilized_cholesky(X, jitter, out=out, verbose=verbose)


def _psd_safe_cholesky(A, out=None, jitter=1e-6, max_tries=32, verbose=True):
    if out is not None:
        out = (out, torch.empty(A.shape[:-2], dtype=torch.int32, device=out.device))

    L, info = torch.linalg.cholesky_ex(A, out=out)
    if not torch.any(info):
        return L

    isnan = torch.isnan(A)
    if isnan.any():
        raise ValueError(
            f"cholesky_cpu: {isnan.sum().item()} of {A.numel()} elements of the {A.shape} tensor are NaN."
        )

    Aprime = A.clone()
    jitter_prev = 0
    for i in range(max_tries):
        jitter_new = jitter * (10**i)
        # add jitter only where needed
        diag_add = (
            ((info > 0) * (jitter_new - jitter_prev))
            .unsqueeze(-1)
            .expand(*Aprime.shape[:-1])
        )
        Aprime.diagonal(dim1=-1, dim2=-2).add_(diag_add)
        jitter_prev = jitter_new
        if verbose:
            print(f"A not p.d., added jitter of {jitter_new:.1e} to the diagonal")
        L, info = torch.linalg.cholesky_ex(Aprime, out=out)
        if not torch.any(info):
            return L
    raise RuntimeError(
        f"Matrix not positive definite after repeatedly adding jitter up to {jitter_new:.1e}."
    )


def psd_safe_cholesky(
    A, upper=False, out=None, jitter=1e-6, max_tries=32, verbose=True
):
    """Compute the Cholesky decomposition of A. If A is only p.s.d, add a small jitter to the diagonal.
    Args:
        A (Tensor):
            The tensor to compute the Cholesky decomposition of
        upper (bool, optional):
            See torch.cholesky
        out (Tensor, optional):
            See torch.cholesky
        jitter (float, optional):
            The jitter to add to the diagonal of A in case A is only p.s.d. If omitted,
            uses settings.cholesky_jitter.value()
        max_tries (int, optional):
            Number of attempts (with successively increasing jitter) to make before raising an error.
    """
    L = _psd_safe_cholesky(
        A, out=out, jitter=jitter, max_tries=max_tries, verbose=verbose
    )
    if upper:
        if out is not None:
            out.transpose_(-1, -2)
        else:
            L = L.transpose(-1, -2)
    return L


if __name__ == "__main__":
    torch.manual_seed(0)
    X = torch.rand(5, 784, 784)
    X = X @ X.mT

    L = psd_safe_cholesky(X, verbose=True)
    print(L)
