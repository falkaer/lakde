import torch

from lakde.utils import InterruptDelay


class ELBOCallback:
    def __init__(self, report_on="iteration", report_every=1, verbose=True):
        self.report_on = report_on
        self.report_every = report_every
        self.verbose = verbose

    def __call__(self, X, model, iter_step):
        N, D = X.shape
        num_blocks = (N - 1) // model.block_size + 1
        batch = iter_step % num_blocks

        if not self.verbose and model.logger is None:
            return  # no reason to waste compute

        if self.report_on == "iteration" or (
            self.report_on == "epoch" and batch + 1 == num_blocks
        ):
            elbo = model.compute_elbo(X)
            if self.verbose:
                print(", ELBO: {:<10f}".format(elbo), end="")


class LikelihoodCallback:
    def __init__(
        self,
        eval_data,
        eval_label="val",
        report_on="epoch",
        report_every=5,
        verbose=True,
        rtol=None,
        save_best=False,
        save_to=None,
    ):
        self.eval_data = eval_data
        self.eval_label = eval_label
        self.report_on = report_on
        self.report_every = report_every
        self.verbose = verbose
        self.save_best = save_best
        self.save_to = save_to
        self.best_ll = None
        self.rtol = rtol

        assert (
            eval_data.is_contiguous()
        ), "Evaluation data must be contiguous, call .contiguous() before fitting"

    def __call__(self, X, model, iter_step):
        N, D = X.shape
        num_blocks = (N - 1) // model.block_size + 1
        epoch = iter_step // num_blocks
        batch = iter_step % num_blocks

        assert (
            self.verbose or model.logger or self.rtol is not None
        ), "Likelihood callback used without printing or logging enabled"

        if self.report_on == "iteration":
            if (batch + 1) % self.report_every != 0:
                return
        elif self.report_on == "epoch":
            if batch + 1 != num_blocks or (epoch + 1) % self.report_every != 0:
                return

        avg_ll = model.log_pred_density(X, self.eval_data).mean()
        if self.verbose:
            print(
                ", {} log likelihood: {:<10f}".format(self.eval_label, avg_ll), end=""
            )

        if model.logger:
            with InterruptDelay():
                model.logger.add_scalar(
                    "log_likelihood/{}".format(self.eval_label),
                    avg_ll,
                    global_step=iter_step,
                )

        if self.save_best:
            if self.best_ll is None or self.best_ll < avg_ll:
                save_to = self.save_to if self.save_to is not None else "model.pt"
                d = model.state_dict()
                d["log_likelihood"] = avg_ll
                torch.save(d, save_to)

        if (
            self.rtol is not None
            and self.best_ll is not None
            and avg_ll - self.best_ll < self.rtol
        ):
            if self.verbose:
                print(
                    "\nReached relative (likelihood) tolerance cutoff at {} iterations".format(
                        iter_step
                    )
                )
            raise StopIteration

        if self.best_ll is None or self.best_ll < avg_ll:
            self.best_ll = avg_ll
