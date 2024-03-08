from lakde import LocalKNearestKDE
from lakde.linalg_util import (
    chol_logdet,
    chol_mm_trace,
    stabilized_cholesky,
    tril_inverse,
)
from lakde.utils import InterruptDelay


class LocalKNearestKDESharedTau(LocalKNearestKDE):
    def calc_posterior(self, block_idx, sigma_0, W_inv_contrib):
        s = block_idx * self.block_size
        e = s + self.block_size
        expect_tau = self.a / self.b
        W_inv_jitter = self.W_inv_jitter[s:e]

        W_inv = sigma_0.to(W_inv_contrib.dtype) * (self.nu_0 * expect_tau)
        W_inv += W_inv_contrib
        W_inv_tril, W_inv_jitter[:] = stabilized_cholesky(
            W_inv, W_inv_jitter, verbose=self.verbose
        )
        W_triu = tril_inverse(W_inv_tril).mT
        return W_triu

    def init_parameters_(self, X, sparsity_threshold=None):
        super().init_parameters_(X, sparsity_threshold)
        N, D = X.shape
        self.a = self.a_0 + N * self.nu_0 * D / 2
        self.b = self.a.clone()  # init so tau = a/b = 1

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

            self.b = self.b_0 + self.nu_0 * self.sigma_0_lambda_mm_trace.sum() / 2
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

    def hparam_state_dict(self):
        d = super().hparam_state_dict()
        d["model"] = "knn_shared_tau"
        return d
