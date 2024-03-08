import math
from typing import Union

import torch

from lakde.kdes import AbstractKDE
from lakde.kernels import calc_W_inv, chol_quad_form, ind2ptr
from lakde.linalg_util import (
    chol_logdet,
    chol_mm_trace,
    lgamma,
    mvdigamma,
    mvlgamma,
    tril_inverse,
    xlogx,
)
from lakde.utils import InterruptDelay, cov, tensor_as


def calc_log_rho_block(X_test, X_train, nu, W_triu, logdet_W):
    D = X_test.size(1)
    expect_logdet_lambda = mvdigamma(nu / 2, D) + D * math.log(2) + logdet_W
    exponent = -0.5 * nu
    const = 0.5 * expect_logdet_lambda
    Q = chol_quad_form(W_triu, X_train, X_test)
    Q *= exponent
    Q += const
    return Q


def expect_log_p_lambda(
    nu_0, expect_logdet_lambda, sigma_0_lambda_mm_trace, logdet_sigma_0, D
):
    expect_log_B_p = -nu_0 / 2 * (
        D * math.log(2) - logdet_sigma_0 - D * torch.log(nu_0)
    ) - mvlgamma(nu_0 / 2, D)
    return (
        expect_log_B_p
        + (nu_0 - D - 1) * expect_logdet_lambda / 2
        - nu_0 * sigma_0_lambda_mm_trace / 2
    )


def expect_log_q_lambda(nu, expect_logdet_lambda, logdet_W, D):
    expect_log_B_q = -nu / 2 * (logdet_W + D * math.log(2)) - mvlgamma(nu / 2, D)
    return expect_log_B_q + (nu - D - 1) * expect_logdet_lambda / 2 - nu / 2 * D


def expect_log_q_z(rnm_log_rnm_contribs):
    return torch.sum(rnm_log_rnm_contribs)


class SharedFullKDE(AbstractKDE):
    def __init__(
        self,
        nu_0: Union[torch.Tensor, float] = None,
        sigma_0: Union[torch.Tensor, None] = None,
        block_size=2048,
        verbose=False,
        logs=False,
    ):
        super().__init__(block_size, verbose, logs)

        # priors
        self.nu_0 = nu_0
        self.sigma_0 = sigma_0
        self.sigma_0_tril = None

        # parameters
        self.W_triu = None
        self.nu = None

        # summaries
        self.partials_ws = None

    def init_parameters_(self, X, sparsity_threshold=None):
        N, D = X.shape
        if self.nu_0 is None:
            self.nu_0 = D
        if self.sigma_0 is None:
            self.sigma_0 = cov(X)

        self.nu_0 = tensor_as(self.nu_0, self.sigma_0)

        if self.nu_0 <= D - 1:
            raise ValueError(
                "nu_0 must be greater than the data dimensionality minus one, or the ELBO will be undefined"
            )

        self.sigma_0_tril = torch.linalg.cholesky(self.sigma_0)
        self.nu = self.nu_0 + N

        W_inv_tril = torch.sqrt(self.nu_0) * self.sigma_0_tril
        self.W_triu = tril_inverse(W_inv_tril).T

        num_blocks = (N - 1) // self.block_size + 1
        self.partials_ws = X.new_empty(num_blocks, D, D)

    def init_summaries_(self, X, sparsity_threshold=None):
        N, D = X.shape
        num_blocks = (N - 1) // self.block_size + 1
        for bj, j in enumerate(range(0, N, self.block_size)):
            self.partials_ws[bj] = self.partial_reestimation_step(
                X, bj, sparsity_threshold=sparsity_threshold
            )
            if self.verbose:
                print("Batch {}/{}...".format(bj + 1, num_blocks))

        W_inv = self.sigma_0.clone()
        W_inv *= self.nu_0
        W_inv += torch.sum(self.partials_ws, dim=0)
        W_inv_tril = torch.linalg.cholesky(W_inv)
        self.W_triu = tril_inverse(W_inv_tril).mT

    def calc_prior_cov(self):
        return self.sigma_0

    def calc_posterior(self):
        W_inv = self.nu_0 * self.calc_prior_cov()
        W_inv += torch.sum(self.partials_ws, dim=0)
        W_inv_tril = torch.linalg.cholesky(W_inv)
        return tril_inverse(W_inv_tril).mT

    def partial_reestimation_step(self, X, bj, sparsity_threshold=None):
        N, D = X.shape
        bsize = self.block_size
        s = bj * bsize
        e = s + bsize
        X_test = X[s:e]
        logdet_W = chol_logdet(self.W_triu)
        rnm_log_consts = X.new_full((X_test.size(0),), -math.inf)
        partial_W_inv = X.new_zeros(D, D)

        # normalization constants change everywhere every time W is updated, recompute them
        for bi, i in enumerate(range(0, N, bsize)):
            X_train = X[i : i + bsize]
            W_triu = self.W_triu.expand(X_train.size(0), D, D)
            log_rho = calc_log_rho_block(X_test, X_train, self.nu, W_triu, logdet_W)
            if bi == bj:  # n = m => rnm = 0
                log_rho.diagonal().fill_(-math.inf)
            torch.logaddexp(
                rnm_log_consts, torch.logsumexp(log_rho, dim=0), out=rnm_log_consts
            )

        # use the normalization constants to normalize responsibilities and compute
        # W_inv contribs on the fly - these are not invalidated when W is updated
        # as they are based only on contributions from rnm
        for bi, i in enumerate(range(0, N, bsize)):
            X_train = X[i : i + bsize]
            W_triu = self.W_triu.expand(X_train.size(0), D, D)
            log_rho = calc_log_rho_block(X_test, X_train, self.nu, W_triu, logdet_W)
            if bi == bj:  # n = m => rnm = 0
                log_rho.diagonal().fill_(-math.inf)
            log_rnm = log_rho.sub_(rnm_log_consts)
            rnm = torch.exp_(log_rnm)
            if sparsity_threshold:
                rnm = torch.threshold_(rnm, sparsity_threshold, 0)
            indices = torch.nonzero(rnm).T.contiguous()
            rnm_values = rnm[indices.unbind()]
            row, col = indices.to(torch.int32)
            # have to use 64 bits for rowptr if numel is too large
            if row.numel() > torch.iinfo(row.dtype).max:
                rowptr = row.new_empty(X_train.size(0) + 1, dtype=torch.int64)
                ind2ptr(row, X_train.size(0), out=rowptr)
            else:
                rowptr = ind2ptr(row, X_train.size(0))
            partial_W_inv += calc_W_inv(rowptr, col, rnm_values, X_train, X_test).sum(
                dim=0
            )

        return partial_W_inv

    def partial_step_(self, X, block_idx, sparsity_threshold=None):
        partial_W_inv = self.partial_reestimation_step(
            X, block_idx, sparsity_threshold=sparsity_threshold
        )
        with InterruptDelay():
            self.partials_ws[block_idx] = partial_W_inv
            self.W_triu = self.calc_posterior()

    def data_log_likelihood_no_rnm(self, X):
        N, D = X.shape
        rnm_log_consts = X.new_full((N,), -math.inf)
        logdet_W = chol_logdet(self.W_triu)
        for bj, j in enumerate(range(0, N, self.block_size)):
            X_test = X[j : j + self.block_size]
            for bi, i in enumerate(range(0, N, self.block_size)):
                X_train = X[i : i + self.block_size]
                W_triu = self.W_triu.expand(X_train.size(0), D, D)
                log_rho = calc_log_rho_block(X_test, X_train, self.nu, W_triu, logdet_W)
                if bi == bj:  # n = m => rnm = 0
                    log_rho.diagonal().fill_(-math.inf)
                torch.logaddexp(
                    rnm_log_consts[j : j + self.block_size],
                    torch.logsumexp(log_rho, dim=0),
                    out=rnm_log_consts[j : j + self.block_size],
                )
        return -D / 2 * math.log(2 * math.pi) + rnm_log_consts

    def data_log_likelihood(self, X):
        N, D = X.shape
        ll = self.data_log_likelihood_no_rnm(X)
        rnm_log_consts = ll + D / 2 * math.log(2 * math.pi)
        logdet_W = chol_logdet(self.W_triu)
        for bj, j in enumerate(range(0, N, self.block_size)):
            X_test = X[j : j + self.block_size]
            for bi, i in enumerate(range(0, N, self.block_size)):
                X_train = X[i : i + self.block_size]
                W_triu = self.W_triu.expand(X_train.size(0), D, D)
                log_rho = calc_log_rho_block(X_test, X_train, self.nu, W_triu, logdet_W)
                if bi == bj:  # n = m => rnm = 0
                    log_rho.diagonal().fill_(-math.inf)
                log_rnm = log_rho.sub_(rnm_log_consts[j : j + self.block_size])
                rnm = torch.exp_(log_rnm)
                rnm_log_rnm = xlogx(rnm, out=rnm)
                ll[j : j + self.block_size] += torch.sum(rnm_log_rnm.sum(dim=0))
        return ll

    def compute_elbo(self, X):
        N, D = X.shape
        logdet_W = chol_logdet(self.W_triu)
        logdet_sigma_0 = chol_logdet(self.sigma_0_tril)
        sigma_0_lambda_mm_trace = self.nu * chol_mm_trace(
            self.sigma_0_tril, self.W_triu
        )
        expect_logdet_lambda = mvdigamma(self.nu / 2, D) + D * math.log(2) + logdet_W

        expect_log_lambda_p = expect_log_p_lambda(
            self.nu_0, expect_logdet_lambda, sigma_0_lambda_mm_trace, logdet_sigma_0, D
        )
        expect_log_lambda_q = expect_log_q_lambda(
            self.nu, expect_logdet_lambda, logdet_W, D
        )

        expect_log_z_p = -N * math.log(N - 1)

        expect_log_likelihood_no_rnm = self.data_log_likelihood_no_rnm(X).sum()
        expect_log_lambda_diff = expect_log_lambda_p - expect_log_lambda_q
        expect_log_z_diff = expect_log_z_p  # - expect_log_z_q

        elbo = expect_log_likelihood_no_rnm + expect_log_lambda_diff + expect_log_z_diff

        if self.logger:
            with InterruptDelay():
                self.logger.add_scalar("elbo/elbo", elbo, self.iter_steps)
                self.logger.add_scalar(
                    "elbo/expect_log_likelihood_no_rnm",
                    expect_log_likelihood_no_rnm,
                    self.iter_steps,
                )
                self.logger.add_scalar(
                    "elbo/expect_log_z_diff", expect_log_z_diff, self.iter_steps
                )
                self.logger.add_scalar(
                    "elbo/expect_log_lambda_diff",
                    expect_log_lambda_diff,
                    self.iter_steps,
                )

                self.logger.add_scalar(
                    "lambda/expect_log_p_lambda", expect_log_lambda_p, self.iter_steps
                )
                self.logger.add_scalar(
                    "lambda/expect_log_q_lambda", expect_log_lambda_q, self.iter_steps
                )

                self.logger.add_scalar(
                    "z/expect_log_p_z", expect_log_z_p, self.iter_steps
                )
                # self.logger.add_scalar('z/expect_log_q_z', expect_log_z_q, self.iter_steps)

        return elbo

    def log_pred_density(self, X, Y):
        N_train, D = X.shape
        N_test = Y.size(0)
        bsize = self.block_size
        logdet_W = chol_logdet(self.W_triu)
        log_coeff = (
            lgamma((self.nu + 1) / 2)
            - lgamma((self.nu + 1 - D) / 2)
            + 1 / 2 * logdet_W
            - D / 2 * math.log(math.pi)
        ).to(self.W_triu.dtype)
        log_densities = Y.new_full((N_test,), -math.inf)
        for bi, i in enumerate(range(0, N_train, bsize)):
            X_train = X[i : i + bsize]
            for j in range(0, N_test, bsize):
                Q = chol_quad_form(
                    self.W_triu.expand(X_train.size(0), D, D), X_train, Y[j : j + bsize]
                )
                log_probs = log_coeff - (self.nu + 1) / 2 * torch.log1p(Q)
                torch.logaddexp(
                    log_densities[j : j + bsize],
                    torch.logsumexp(log_probs, dim=0),
                    out=log_densities[j : j + bsize],
                )
        return -math.log(N_train) + log_densities

    def sample(self, X, n):
        N, D = X.shape
        eps = torch.finfo(X.dtype).eps
        scale_tril = tril_inverse(
            torch.sqrt(self.nu + 1 - D)[:, None, None] * self.W_triu
        ).T
        inds = torch.randint(N, (n,), device=X.device)

        # x_hat ~ St(x_n, (nu + 1 - D) * W, nu + 1 - D) is equivalent to
        # x_hat = x_n + y / sqrt(u / (nu + 1 - D)) where y ~ N(0, ((nu + 1 - D) W)^-1) and u ~ Chi^2(nu + 1 - D)
        # threshold u to avoid very unlikely divide by zero
        y = torch.distributions.MultivariateNormal(
            loc=X.new_zeros(D), scale_tril=scale_tril
        ).sample((n,))
        u = torch.clamp_min(
            torch.distributions.Chi2(df=self.nu + 1 - D).sample((n,)), eps
        )
        return X[inds] + y / torch.sqrt(u / (self.nu + 1 - D))[..., None]

    def load_state_dict(self, state_dict):
        self.nu_0 = state_dict["nu_0"]
        self.sigma_0 = state_dict["sigma_0"]

        self.nu = state_dict["nu"]
        self.W_triu = state_dict["W_triu"]
        self.partials_ws = state_dict["partial_W_invs"]

    def state_dict(self):
        d = super().state_dict()
        d.update(
            {
                "nu_0": self.nu_0,
                "sigma_0": self.sigma_0,
                "nu": self.nu,
                "W_triu": self.W_triu,
                "partial_W_invs": self.partials_ws,
                "block_size": self.block_size,
                "verbose": self.verbose,
                "iter_steps": self.iter_steps,
            }
        )
        return d

    def hparam_state_dict(self):
        d = super().hparam_state_dict()
        d.update({"model": "full"})
        return d
