import math
from typing import Union

import torch

from lakde.kdes import AbstractKDE
from lakde.kernels import calc_W_inv, chol_quad_form, ind2ptr
from lakde.linalg_util import lgamma, xlogx
from lakde.utils import InterruptDelay, tensor_as, var


def calc_log_rho_block(X_test, X_train, nu, w):
    D = X_test.size(1)
    expect_log_lambda = D * (torch.digamma(nu) - torch.log(w))
    W_triu = torch.diag_embed(torch.sqrt(1 / w).expand(D)).expand(X_train.size(0), D, D)
    exponent = -0.5 * nu
    const = 0.5 * expect_log_lambda
    Q = chol_quad_form(W_triu, X_train, X_test)
    Q *= exponent
    Q += const
    return Q


def expect_log_p_lambda(nu_0, expect_lambda, expect_log_lambda, sigma_sq):
    return (
        -torch.lgamma(nu_0)
        + nu_0 * (torch.log(nu_0) + torch.log(sigma_sq))
        + (nu_0 - 1) * expect_log_lambda
        - expect_lambda / sigma_sq
    )


def expect_log_q_lambda(nu, w, expect_log_lambda):
    return -torch.lgamma(nu) + nu * torch.log(w) + (nu - 1) * expect_log_lambda - nu


def expect_log_q_z(rnm_log_rnm_contribs):
    return torch.sum(rnm_log_rnm_contribs)


class SharedScalarKDE(AbstractKDE):
    def __init__(
        self,
        nu_0: Union[torch.Tensor, float] = None,
        sigma_0_sq: Union[torch.Tensor, None] = None,
        block_size=2048,
        verbose=False,
        logs=False,
    ):
        super().__init__(block_size, verbose, logs)

        # priors
        self.sigma_0_sq = sigma_0_sq
        self.nu_0 = nu_0

        # parameters
        self.nu = None
        self.w = None

        # summaries
        self.partial_ws = None

    def init_parameters_(self, X, sparsity_threshold=None):
        N, D = X.shape
        if self.nu_0 is None:
            self.nu_0 = D
        if self.sigma_0_sq is None:
            self.sigma_0_sq = var(X).mean()

        self.nu_0 = tensor_as(self.nu_0, self.sigma_0_sq)

        if self.nu_0 <= D - 1:
            raise ValueError(
                "nu_0 must be greater than the data dimensionality minus one, or the ELBO will be undefined"
            )

        # init such that E[lambda^2] = nu / w = 1 / sigma_0_sq
        self.nu = self.nu_0 + N * D / 2
        self.w = self.nu * self.sigma_0_sq

        num_blocks = (N - 1) // self.block_size + 1
        self.partial_ws = X.new_empty(num_blocks)

    def init_summaries_(self, X, sparsity_threshold=None):
        N, D = X.shape
        num_blocks = (N - 1) // self.block_size + 1
        for bj, j in enumerate(range(0, N, self.block_size)):
            self.partial_ws[bj] = self._partial_reestimation_step(
                X, bj, sparsity_threshold=sparsity_threshold
            )
            if self.verbose:
                print("Batch {}/{}...".format(bj + 1, num_blocks))
        self.w = self._calc_posterior()

    def _calc_posterior(self):
        return self.nu_0 * self.sigma_0_sq + torch.sum(self.partial_ws) / 2

    def _partial_reestimation_step(self, X, bj, sparsity_threshold=None):
        N, D = X.shape
        bsize = self.block_size
        s = bj * bsize
        e = s + bsize
        X_test = X[s:e]
        rnm_log_consts = X.new_full((X_test.size(0),), -math.inf)
        partial_w = X.new_zeros(())

        # normalization constants change everywhere every time W is updated, recompute them
        for bi, i in enumerate(range(0, N, bsize)):
            X_train = X[i : i + bsize]
            log_rho = calc_log_rho_block(X_test, X_train, self.nu, self.w)
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
            log_rho = calc_log_rho_block(X_test, X_train, self.nu, self.w)
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
            partial_w += (
                calc_W_inv(rowptr, col, rnm_values, X_train, X_test)
                .diagonal(dim1=-1, dim2=-2)
                .sum()
            )

        return partial_w

    def partial_step_(self, X, block_idx, sparsity_threshold=None):
        partial_w = self._partial_reestimation_step(
            X, block_idx, sparsity_threshold=sparsity_threshold
        )
        with InterruptDelay():
            self.partial_ws[block_idx] = partial_w
            self.w = self._calc_posterior()

    def data_log_likelihood_no_rnm(self, X):
        N, D = X.shape
        rnm_log_consts = X.new_full((N,), -math.inf)
        for bj, j in enumerate(range(0, N, self.block_size)):
            X_test = X[j : j + self.block_size]
            for bi, i in enumerate(range(0, N, self.block_size)):
                X_train = X[i : i + self.block_size]
                log_rho = calc_log_rho_block(X_test, X_train, self.nu, self.w)
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
        for bj, j in enumerate(range(0, N, self.block_size)):
            X_test = X[j : j + self.block_size]
            for bi, i in enumerate(range(0, N, self.block_size)):
                X_train = X[i : i + self.block_size]
                log_rho = calc_log_rho_block(X_test, X_train, self.nu, self.w)
                if bi == bj:  # n = m => rnm = 0
                    log_rho.diagonal().fill_(-math.inf)
                log_rnm = log_rho.sub_(rnm_log_consts[j : j + self.block_size])
                rnm = torch.exp_(log_rnm)
                rnm_log_rnm = xlogx(rnm, out=rnm)
                ll[j : j + self.block_size] += torch.sum(rnm_log_rnm.sum(dim=0))
        return ll

    def compute_elbo(self, X):
        N, D = X.shape
        expect_lambda = self.nu / self.w
        expect_log_lambda = torch.digamma(self.nu) - torch.log(self.w)

        expect_log_lambda_p = expect_log_p_lambda(
            self.nu_0, expect_lambda, expect_log_lambda, self.sigma_0_sq
        )
        expect_log_lambda_q = expect_log_q_lambda(self.nu, self.w, expect_log_lambda)
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
        W_triu = torch.diag_embed((1 / torch.sqrt(2 * self.w)).expand(D))
        log_densities = Y.new_full((N_test,), -math.inf)
        log_coeff = (
            lgamma(self.nu + D / 2)
            - lgamma(self.nu)
            - D / 2 * (torch.log(self.w) + math.log(2 * math.pi))
        )
        for bi, i in enumerate(range(0, N_train, self.block_size)):
            X_train = X[i : i + self.block_size]
            for j in range(0, N_test, self.block_size):
                Q = chol_quad_form(
                    W_triu.expand(X_train.size(0), D, D),
                    X_train,
                    Y[j : j + self.block_size],
                )
                log_probs = log_coeff - (self.nu + D / 2) * torch.log1p(Q)
                torch.logaddexp(
                    log_densities[j : j + self.block_size],
                    torch.logsumexp(log_probs, dim=0),
                    out=log_densities[j : j + self.block_size],
                )
        return -math.log(N_train) + log_densities

    def sample(self, X, n):
        N, D = X.shape
        eps = torch.finfo(X.dtype).eps
        scale_tril = torch.diag_embed(torch.sqrt(self.w / self.nu).expand(D))
        inds = torch.randint(N, (n,), device=X.device)

        # x_hat ~ St(x_n, diag(sqrt(nu / w)), 2nu) is equivalent to
        # x_hat = x_n + y / sqrt(u / (2nu)) where y ~ N(0, (diag(sqrt(nu / w)))^-1) and u ~ Chi^2(2nu)
        # threshold u to avoid very unlikely divide by zero
        y = torch.distributions.MultivariateNormal(
            loc=X.new_zeros(D), scale_tril=scale_tril
        ).sample((n,))
        u = torch.clamp_min(torch.distributions.Chi2(df=2 * self.nu).sample((n,)), eps)
        return X[inds] + y / torch.sqrt(u / (2 * self.nu))[..., None]

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.nu_0 = state_dict["nu_0"]
        self.sigma_0_sq = state_dict["sigma_0_sq"]

        self.nu = state_dict["nu"]
        self.w = state_dict["w"]
        self.partial_ws = state_dict["partial_ws"]

    def state_dict(self):
        d = super().state_dict()
        d.update(
            {
                "nu_0": self.nu_0,
                "sigma_0_sq": self.sigma_0_sq,
                "nu": self.nu,
                "w": self.w,
                "partial_ws": self.partial_ws,
            }
        )
        return d

    def hparam_state_dict(self):
        d = super().hparam_state_dict()
        d.update({"model": "scalar"})
        return d
