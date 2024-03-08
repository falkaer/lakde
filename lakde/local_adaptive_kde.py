import math
from abc import ABC, abstractmethod

import torch

from lakde.kdes import AbstractKDE
from lakde.kernels import calc_W_inv, chol_quad_form
from lakde.linalg_util import (
    chol_logdet,
    lgamma,
    logsubexp,
    mvdigamma,
    stabilized_cholesky,
    tril_inverse,
    xlogx,
)
from lakde.sparse_util import cat_sparse, ind2ptr
from lakde.utils import sample_multinomial


def expect_log_p_tau(a_0, b_0, expect_tau, expect_log_tau):
    return torch.sum(
        -lgamma(a_0)
        + a_0 * torch.log(b_0)
        + (a_0 - 1) * expect_log_tau
        - b_0 * expect_tau
    )


def expect_log_q_tau(a, b, expect_tau, expect_log_tau):
    return torch.sum(
        -lgamma(a) + a * torch.log(b) + (a - 1) * expect_log_tau - b * expect_tau
    )


def calc_log_rho_block(X_test, X_train, nu, W_triu, logdet_W):
    D = X_test.size(-1)
    expect_logdet_lambda = mvdigamma(nu / 2, D) + D * math.log(2) + logdet_W
    exponent = -0.5 * nu
    const = 0.5 * expect_logdet_lambda
    Q = chol_quad_form(W_triu, X_train, X_test)
    Q *= exponent[:, None]
    Q += const[:, None]
    return Q


class LocallyAdaptiveKDE(AbstractKDE, ABC):
    def __init__(self, nu_0, a_0, b_0, block_size, verbose, logs):
        super().__init__(block_size, verbose, logs)
        self.nu_0 = nu_0
        self.nu = None

        self.a_0 = a_0
        self.b_0 = b_0

        self.a = None
        self.b = None

        self.rnm_blocks = None
        self.rnm_log_consts = None
        self.logdet_W = None

        self.rnm_active_contribs = None
        self.W_inv_jitter = None

    def calc_posterior_contribs(self, X, block_idx):
        s = block_idx * self.block_size
        e = s + self.block_size
        X_train = X[s:e]
        indices, values = self.rnm_blocks[block_idx]
        row, col = indices
        # have to use 64 bits for rowptr if numel is too large TODO: can this happen?
        if row.numel() > torch.iinfo(row.dtype).max:
            rowptr = row.new_empty(X_train.size(0) + 1, dtype=torch.int64)
            ind2ptr(row, X_train.size(0), out=rowptr)
        else:
            rowptr = ind2ptr(row, X_train.size(0))
        W_inv_contrib = calc_W_inv(rowptr, col, values, X_train, X)
        # nu = nu_0 + sum(rnm, dim=1)
        nu = self.nu_0.expand(X_train.size(0)).scatter_add(0, row.long(), values)
        return W_inv_contrib, nu

    def calc_posterior(self, block_idx, sigma_0, W_inv_contrib):
        s = block_idx * self.block_size
        e = s + self.block_size
        expect_tau = self.a[s:e] / self.b[s:e]
        W_inv_jitter = self.W_inv_jitter[s:e]
        W_inv = (
            sigma_0.to(W_inv_contrib.dtype) * (self.nu_0 * expect_tau)[:, None, None]
        )
        W_inv += W_inv_contrib
        W_inv_tril, W_inv_jitter[:] = stabilized_cholesky(
            W_inv, W_inv_jitter, verbose=self.verbose
        )
        W_triu = tril_inverse(W_inv_tril).mT
        return W_triu

    def partial_expectation_step(
        self,
        X,
        block_idx,
        W_triu,
        nu,
        update_normalization=True,
        sparsity_threshold=None,
    ):
        N, D = X.shape
        s = block_idx * self.block_size
        e = s + self.block_size
        X_train = X[s:e]
        logdet_W = chol_logdet(W_triu)

        rnm_log_consts = self.rnm_log_consts
        old_log_consts = rnm_log_consts
        rnm_blocks = []

        if update_normalization:
            rnm_log_consts = rnm_log_consts.clone()
            indices, values = self.rnm_blocks[block_idx]
            old_rnm_sums = X.new_zeros(N).scatter_add_(0, indices[1].long(), values)
            old_log_const_contribs = torch.log(old_rnm_sums) + old_log_consts

        for bj, j in enumerate(range(0, N, self.block_size)):
            X_test = X[j : j + self.block_size]
            log_rho = calc_log_rho_block(X_test, X_train, nu, W_triu, logdet_W)
            if block_idx == bj:  # n = m => rnm = 0
                log_rho.diagonal().fill_(-math.inf)
            if update_normalization:
                new_log_const_contribs = torch.logsumexp(log_rho, dim=0)
                torch.logaddexp(
                    logsubexp(
                        old_log_consts[j : j + self.block_size],
                        old_log_const_contribs[j : j + self.block_size],
                    ),
                    new_log_const_contribs,
                    out=rnm_log_consts[j : j + self.block_size],
                )
            log_rnm = log_rho.sub_(rnm_log_consts[j : j + self.block_size])
            rnm = torch.exp_(log_rnm)
            if sparsity_threshold:
                rnm = torch.threshold_(rnm, sparsity_threshold, 0)
            indices = torch.nonzero(rnm).T.contiguous()
            values = rnm[indices.unbind()]
            indices = indices.to(torch.int32)
            rnm_blocks.append((indices, values))

        # no values overlap so don't need full coalesce, and only need row to be sorted
        indices, values = cat_sparse(*zip(*rnm_blocks), self.block_size, dim=1)
        perm = torch.argsort(indices[0])
        indices, values = indices[:, perm], values[perm]
        if update_normalization:
            return indices, values, rnm_log_consts
        else:
            return indices, values

    def rebalance_responsibilities_(self, X, block_idx, rnm_log_consts):
        N = X.size(0)
        finfo = torch.finfo(X.dtype)
        coeff = torch.exp(self.rnm_log_consts - rnm_log_consts)
        coeff = torch.clamp_(coeff, min=finfo.tiny, max=finfo.max)

        # renormalize other blocks and sum them to enforce sum(rnm, dim=0) = 1
        rnm_sums = X.new_zeros(N)
        for bi, i in enumerate(range(0, N, self.block_size)):
            indices, values = self.rnm_blocks[bi]
            col = indices[1].long()
            if bi != block_idx:
                values *= coeff[col]
            rnm_sums.scatter_add_(0, col, values)

        # correct normalization constant drift (enforce sum(rnm, dim=0) = 1)
        rnm_sums = torch.clamp_min(rnm_sums, finfo.eps)
        rnm_log_consts += rnm_sums.log()
        coeff = torch.reciprocal(rnm_sums)
        torch.clamp_(coeff, min=finfo.tiny, max=finfo.max)

        for bi, i in enumerate(range(0, N, self.block_size)):
            indices, values = self.rnm_blocks[bi]
            values *= coeff[indices[1].long()]

        return rnm_log_consts

    def data_log_likelihood_no_rnm(self, X):
        N, D = X.shape
        return -D / 2 * math.log(2 * math.pi) + self.rnm_log_consts

    def data_log_likelihood(self, X):
        N, D = X.shape
        ll = self.data_log_likelihood_no_rnm(X)
        for bi, i in enumerate(range(0, N, self.block_size)):
            indices, values = self.rnm_blocks[bi]
            rnm_log_rnm = xlogx(values)
            ll.scatter_add_(0, indices[1].long(), rnm_log_rnm)
        return ll

    @abstractmethod
    def calc_prior_cov(self, X, bi):
        pass

    def log_pred_density(self, X, Y):
        N_train, D = X.shape
        N_test = Y.size(0)
        log_densities = Y.new_full((N_test,), -math.inf)
        for bi, i in enumerate(range(0, N_train, self.block_size)):
            X_train = X[i : i + self.block_size]
            W_inv_contrib, nu = self.calc_posterior_contribs(X, bi)
            sigma_0, _ = self.calc_prior_cov(X, bi)
            W_triu = self.calc_posterior(bi, sigma_0, W_inv_contrib)
            log_coeff = (
                lgamma((nu + 1) / 2)
                - lgamma((nu + 1 - D) / 2)
                + 1 / 2 * chol_logdet(W_triu)
                - D / 2 * math.log(math.pi)
            ).to(W_triu.dtype)
            for j in range(0, N_test, self.block_size):
                Q = chol_quad_form(W_triu, X_train, Y[j : j + self.block_size])
                log_probs = log_coeff[:, None] - (nu[:, None] + 1) / 2 * torch.log1p(Q)
                torch.logaddexp(
                    log_densities[j : j + self.block_size],
                    torch.logsumexp(log_probs, dim=0),
                    out=log_densities[j : j + self.block_size],
                )
        return -math.log(N_train) + log_densities

    def sample(self, X, n):
        N, D = X.shape
        block_sizes = []
        for i in range(N, 0, -self.block_size):
            block_sizes.append(min(i, self.block_size))

        probs = torch.tensor(block_sizes, dtype=X.dtype, device=X.device)
        probs = probs / probs.sum()
        samples_per_batch = sample_multinomial(n, probs)
        out = X.new_empty(n, D)
        eps = torch.finfo(X.dtype).eps
        offset = 0

        for bi, num_samples in enumerate(samples_per_batch):
            s = bi * self.block_size
            e = s + self.block_size
            X_train = X[s:e]

            sigma_0, _ = self.calc_prior_cov(X, bi)
            W_inv_contrib, nu = self.calc_posterior_contribs(X, bi)
            W_triu = self.calc_posterior(bi, sigma_0, W_inv_contrib)
            scale_tril = tril_inverse(torch.sqrt(nu + 1 - D)[:, None, None] * W_triu).mT

            # select the points from which to draw
            for i in range(0, num_samples, self.block_size):
                sample_size = min(num_samples - i, self.block_size)
                inds = torch.randint(block_sizes[bi], (sample_size,), device=X.device)
                X_train_i = X_train[inds]
                scale_tril_i = scale_tril[inds]
                nu_i = nu[inds]

                # x_hat ~ St(x_n, (nu + 1 - D) * W, nu + 1 - D) is equivalent to
                # x_hat = x_n + y / sqrt(u / (nu + 1 - D)) where y ~ N(0, ((nu + 1 - D) W)^-1) and u ~ Chi^2(nu + 1 - D)
                # threshold u to avoid very unlikely divide by zero
                y = torch.distributions.MultivariateNormal(
                    loc=torch.zeros_like(X_train_i), scale_tril=scale_tril_i
                ).sample()
                u = torch.clamp_min(
                    torch.distributions.Chi2(df=nu_i + 1 - D).sample(), eps
                )
                out[offset : offset + sample_size] = (
                    X_train_i + y / torch.sqrt(u / (nu_i + 1 - D))[..., None]
                )
                offset += sample_size

        return out

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)

        self.nu_0 = state_dict["nu_0"]
        self.nu = state_dict["nu"]

        self.a_0 = state_dict["a_0"]
        self.b_0 = state_dict["b_0"]

        self.a = state_dict["a"]
        self.b = state_dict["b"]

        self.rnm_blocks = state_dict["rnm_blocks"]
        self.rnm_log_consts = state_dict["rnm_log_consts"]

        self.logdet_W = state_dict["logdet_W"]
        self.rnm_active_contribs = state_dict["rnm_active_contribs"]

        self.W_inv_jitter = state_dict["W_inv_jitter"]

    def state_dict(self):
        d = super().state_dict()
        d.update(
            {
                "nu_0": self.nu_0,
                "nu": self.nu,
                "a_0": self.a_0,
                "b_0": self.b_0,
                "a": self.a,
                "b": self.b,
                "rnm_blocks": self.rnm_blocks,
                "rnm_log_consts": self.rnm_log_consts,
                "logdet_W": self.logdet_W,
                "rnm_active_contribs": self.rnm_active_contribs,
                "W_inv_jitter": self.W_inv_jitter,
            }
        )
        return d

    def hparam_state_dict(self):
        d = super().hparam_state_dict()
        d.update({"nu_0": self.nu_0.item()})
        return d
