import math
from typing import Union

import numpy as np
import torch

from lakde.linalg_util import (
    chol_logdet,
    chol_mm_trace,
    mvdigamma,
    mvlgamma,
    stabilized_cholesky,
)
from lakde.local_adaptive_kde import (
    LocallyAdaptiveKDE,
    calc_log_rho_block,
    expect_log_p_tau,
    expect_log_q_tau,
)
from lakde.sparse_util import graph_to_slices
from lakde.utils import InterruptDelay, batched_knn_cov, compute_knn_graph, tensor_as


def expect_log_p_lambda(
    nu_0,
    expect_tau,
    expect_log_tau,
    expect_logdet_lambda,
    sigma_0_lambda_mm_trace,
    logdet_sigma_0,
    D,
):
    expect_log_B_p = -nu_0 / 2 * (
        D * math.log(2) - logdet_sigma_0 - D * expect_log_tau - D * torch.log(nu_0)
    ) - mvlgamma(nu_0 / 2, D)
    return torch.sum(
        expect_log_B_p
        + (nu_0 - D - 1) * expect_logdet_lambda / 2
        - expect_tau * nu_0 * sigma_0_lambda_mm_trace / 2
    )


def expect_log_q_lambda(nu, expect_logdet_lambda, logdet_W, D):
    expect_log_B_q = -nu / 2 * (logdet_W + D * math.log(2)) - mvlgamma(nu / 2, D)
    return torch.sum(
        expect_log_B_q + (nu - D - 1) * expect_logdet_lambda / 2 - nu / 2 * D
    )


class LocalKNearestKDE(LocallyAdaptiveKDE):
    def __init__(
        self,
        nu_0: Union[torch.Tensor, float] = None,
        k_or_knn_graph: Union[torch.Tensor, int] = None,
        block_size: int = 512,
        a_0: Union[torch.Tensor, float] = 1e-6,
        b_0: Union[torch.Tensor, float] = 1e-6,
        verbose=False,
        logs=False,
    ):
        super().__init__(nu_0, a_0, b_0, block_size, verbose, logs)
        self.k_or_knn_graph = k_or_knn_graph
        self.k = None

        self.rnm_blocks = None
        self.knn_graph = None

        self.logdet_sigma_0 = None
        self.sigma_0_lambda_mm_trace = None

        self.sigma_0_jitter = None

    def init_parameters_(self, X, sparsity_threshold=None):
        N, D = X.shape
        self.block_size = min(self.block_size, N)
        self.sigma_0_jitter = X.new_zeros(N)
        self.W_inv_jitter = X.new_zeros(N)

        self.a_0 = tensor_as(self.a_0, X)
        self.b_0 = tensor_as(self.b_0, X)

        if self.nu_0 is not None:
            self.nu_0 = tensor_as(self.nu_0, X)
        else:
            self.nu_0 = X.new_full((), D)

        if self.nu_0.item() <= D - 1:
            raise ValueError(
                "nu_0 must be greater than the data dimensionality minus one, or the ELBO will be undefined"
            )

        if isinstance(self.k_or_knn_graph, int):
            if self.verbose:
                print("Computing nearest neighbour graph...")
            self.knn_graph = compute_knn_graph(
                X,
                self.k_or_knn_graph,
                1024,
                include_self=False,
                metric="l2",
                dtype=torch.int32,
            )
        elif isinstance(self.k_or_knn_graph, np.ndarray):
            self.k_or_knn_graph = torch.from_numpy(self.k_or_knn_graph).to(X.device)
        elif self.k_or_knn_graph.dtype in [torch.int, torch.long]:
            self.knn_graph = self.k_or_knn_graph.to(X.device)
        else:
            raise ValueError("k_or_knn_graph type not recognised")
        del self.k_or_knn_graph

        self.k = self.knn_graph.size(-1)

        if sparsity_threshold:  # truncate the graph to ensure sparsity threshold
            init_k = min(math.floor(1 / sparsity_threshold), self.knn_graph.size(-1))
            init_graph = self.knn_graph[:, :init_k]
        else:
            init_graph = self.knn_graph
            init_k = self.k

        num_blocks = (N - 1) // self.block_size + 1
        self.rnm_blocks = []
        inds = graph_to_slices(init_graph, self.block_size)

        for i in range(num_blocks):
            self.rnm_blocks.append(
                (inds[i], X.new_full((inds[i].size(1),), 1 / init_k))
            )

        self.nu = X.new_empty(N)

        self.a = (self.a_0 + self.nu_0 * D / 2).expand(N)
        self.b = self.a.clone()  # init so tau = a/b = 1

        self.rnm_active_contribs = X.new_zeros(())
        self.rnm_log_consts = X.new_full((N,), -math.inf)
        self.logdet_sigma_0 = X.new_empty(N)
        self.logdet_W = X.new_empty(N)
        self.sigma_0_lambda_mm_trace = X.new_empty(N)

    def init_summaries_(self, X, sparsity_threshold=None):
        N = X.size(0)
        num_blocks = (N - 1) // self.block_size + 1
        if self.verbose:
            print("Computing normalization constants...")
        for bi, i in enumerate(range(0, N, self.block_size)):
            X_train = X[i : i + self.block_size]
            sigma_0, sigma_0_tril = self.calc_prior_cov(X, bi)
            W_inv_contrib, nu = self.calc_posterior_contribs(X, bi)
            W_triu = self.calc_posterior(bi, sigma_0, W_inv_contrib)
            logdet_W = chol_logdet(W_triu)
            self.nu[i : i + self.block_size] = nu
            self.logdet_W[i : i + self.block_size] = logdet_W
            self.logdet_sigma_0[i : i + self.block_size] = chol_logdet(sigma_0_tril)
            self.sigma_0_lambda_mm_trace[i : i + self.block_size] = nu * chol_mm_trace(
                sigma_0_tril, W_triu
            )
            for bj, j in enumerate(range(0, N, self.block_size)):
                X_test = X[j : j + self.block_size]
                log_rho = calc_log_rho_block(X_test, X_train, nu, W_triu, logdet_W)
                if bi == bj:  # n = m => rnm = 0
                    log_rho.diagonal().fill_(-math.inf)
                torch.logaddexp(
                    self.rnm_log_consts[j : j + self.block_size],
                    torch.logsumexp(log_rho, dim=0),
                    out=self.rnm_log_consts[j : j + self.block_size],
                )
            if self.verbose:
                print("Batch {}/{}...".format(bi + 1, num_blocks))

        if self.verbose:
            print("Computing normalized responsibilities...")
        for bi, i in enumerate(range(0, N, self.block_size)):
            sigma_0, sigma_0_tril = self.calc_prior_cov(X, bi)
            W_inv_contrib, nu = self.calc_posterior_contribs(X, bi)
            W_triu = self.calc_posterior(bi, sigma_0, W_inv_contrib)
            self.rnm_blocks[bi] = self.partial_expectation_step(
                X,
                bi,
                W_triu,
                nu,
                update_normalization=False,
                sparsity_threshold=sparsity_threshold,
            )
            self.rnm_active_contribs += self.rnm_blocks[bi][1].numel()
            if self.verbose:
                print("Batch {}/{}...".format(bi + 1, num_blocks))

        if self.verbose:
            print("Correcting normalization constant drift...")

        self.rnm_log_consts = self.rebalance_responsibilities_(
            X, -1, self.rnm_log_consts
        )

    def calc_prior_cov(self, X, block_idx):
        s = block_idx * self.block_size
        e = s + self.block_size
        sigma_0_jitter = self.sigma_0_jitter[s:e]
        knn_graph = self.knn_graph[s:e]
        sigma_0 = batched_knn_cov(
            X, X[s:e], knn_graph.long(), include_self=False, self_center=True
        )
        sigma_0_tril, sigma_0_jitter[:] = stabilized_cholesky(
            sigma_0, sigma_0_jitter, verbose=self.verbose
        )
        sigma_0.diagonal(dim1=-2, dim2=-1).add_(sigma_0_jitter[..., None])
        return sigma_0, sigma_0_tril

    def partial_step_(self, X, block_idx, sparsity_threshold=None):
        N, D = X.shape
        s = block_idx * self.block_size
        e = s + self.block_size

        sigma_0, sigma_0_tril = self.calc_prior_cov(X, block_idx)
        W_inv_contrib, nu = self.calc_posterior_contribs(X, block_idx)
        W_triu = self.calc_posterior(block_idx, sigma_0, W_inv_contrib)
        rnm_indices, rnm_values, rnm_log_consts = self.partial_expectation_step(
            X, block_idx, W_triu, nu, sparsity_threshold=sparsity_threshold
        )
        logdet_W = chol_logdet(W_triu)
        logdet_sigma_0 = chol_logdet(sigma_0_tril)
        sigma_0_lambda_mm_trace = nu * chol_mm_trace(sigma_0_tril, W_triu)

        with InterruptDelay():
            self.rnm_active_contribs -= self.rnm_blocks[block_idx][1].numel()
            self.rnm_active_contribs += rnm_values.numel()
            self.rnm_blocks[block_idx] = (
                rnm_indices.to(self.knn_graph.dtype),
                rnm_values,
            )
            self.logdet_W[s:e] = logdet_W
            self.logdet_sigma_0[s:e] = logdet_sigma_0
            self.sigma_0_lambda_mm_trace[s:e] = sigma_0_lambda_mm_trace
            self.nu[s:e] = nu

            self.b[s:e] = self.b_0 + self.nu_0 * sigma_0_lambda_mm_trace / 2
            self.rnm_log_consts = self.rebalance_responsibilities_(
                X, block_idx, rnm_log_consts
            )

            if self.logger:
                self.logger.add_scalar(
                    "z/active_rnm_proportion",
                    self.rnm_active_contribs / N**2,
                    self.iter_steps,
                )
                self.logger.add_scalar(
                    "z/active_rnm_per_point",
                    self.rnm_active_contribs / N,
                    self.iter_steps,
                )
                self.logger.add_scalar(
                    "z/active_rnm_total", self.rnm_active_contribs, self.iter_steps
                )

    def compute_elbo(self, X):
        N, D = X.shape
        expect_tau = self.a / self.b
        expect_log_tau = torch.digamma(self.a) - torch.log(self.b)
        expect_logdet_lambda = (
            mvdigamma(self.nu / 2, D) + D * math.log(2) + self.logdet_W
        )

        expect_log_tau_p = expect_log_p_tau(
            self.a_0, self.b_0, expect_tau, expect_log_tau
        )
        expect_log_tau_q = expect_log_q_tau(self.a, self.b, expect_tau, expect_log_tau)

        expect_log_lambda_p = expect_log_p_lambda(
            self.nu_0,
            expect_tau,
            expect_log_tau,
            expect_logdet_lambda,
            self.sigma_0_lambda_mm_trace,
            self.logdet_sigma_0,
            D,
        )
        expect_log_lambda_q = expect_log_q_lambda(
            self.nu, expect_logdet_lambda, self.logdet_W, D
        )

        expect_log_z_p = -N * math.log(N - 1)

        expect_log_likelihood_no_rnm = self.data_log_likelihood_no_rnm(X).sum()
        expect_log_lambda_diff = expect_log_lambda_p - expect_log_lambda_q
        expect_log_z_diff = expect_log_z_p  # - expect_log_z_q
        expect_log_tau_diff = expect_log_tau_p - expect_log_tau_q

        elbo = (
            expect_log_likelihood_no_rnm
            + expect_log_lambda_diff
            + expect_log_z_diff
            + expect_log_tau_diff
        )

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
                    "elbo/expect_log_tau_diff", expect_log_tau_diff, self.iter_steps
                )

                self.logger.add_scalar(
                    "lambda/expect_log_p_lambda", expect_log_lambda_p, self.iter_steps
                )
                self.logger.add_scalar(
                    "lambda/expect_log_q_lambda", expect_log_lambda_q, self.iter_steps
                )

                self.logger.add_scalar(
                    "tau/expect_log_p_tau", expect_log_tau_p, self.iter_steps
                )
                self.logger.add_scalar(
                    "tau/expect_log_q_tau", expect_log_tau_q, self.iter_steps
                )

                self.logger.add_scalar("tau/a_avg", torch.mean(self.a), self.iter_steps)
                self.logger.add_scalar("tau/b_avg", torch.mean(self.b), self.iter_steps)
                self.logger.add_scalar(
                    "tau/tau_avg", torch.mean(self.a / self.b), self.iter_steps
                )

                self.logger.add_scalar(
                    "z/expect_log_p_z", expect_log_z_p, self.iter_steps
                )
                # self.logger.add_scalar('z/expect_log_q_z', expect_log_z_q, self.iter_steps)

        return elbo

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        # parameters
        self.knn_graph = state_dict["knn_graph"]
        self.k = state_dict["k"]

        # summaries
        self.logdet_sigma_0 = state_dict["logdet_sigma_0"]
        self.sigma_0_lambda_mm_trace = state_dict["sigma_0_lambda_mm_trace"]

        # implementation parameters
        self.sigma_0_jitter = state_dict["sigma_0_jitter"]

    def state_dict(self):
        d = super().state_dict()
        d.update(
            {
                "knn_graph": self.knn_graph,
                "k": self.k,
                "logdet_sigma_0": self.logdet_sigma_0,
                "sigma_0_lambda_mm_trace": self.sigma_0_lambda_mm_trace,
                "sigma_0_jitter": self.sigma_0_jitter,
            }
        )
        return d

    def hparam_state_dict(self):
        d = super().hparam_state_dict()
        d.update({"k": self.k, "model": "knn"})
        return d
