import argparse
import math

import numpy as np
import torch
from datasets import *
from lakde.kernels import chol_quad_form
from scipy.stats import gaussian_kde
from tqdm import tqdm


# rewrite of the logpdf from scipy that actually
# finishes before the heat death of the universe
def torch_logpdf(model, Y, bsize):
    X = torch.from_numpy(model.dataset.T).to(Y.device)
    N, D = X.shape
    M = Y.size(0)

    result = Y.new_full((M,), -math.inf)
    logw = torch.from_numpy(model.weights).log().unsqueeze(dim=1).to(Y.device)
    inv_cov = torch.from_numpy(model.inv_cov).to(Y.device).float()
    inv_cov_tril = torch.linalg.cholesky(inv_cov)
    for i in tqdm(range(0, N, bsize)):
        for j in range(0, M, bsize):
            Xt = X[i : i + bsize]
            Yt = Y[j : j + bsize]
            energy = chol_quad_form(inv_cov_tril.expand(Xt.size(0), D, D), Xt, Yt)
            log_likelihood = logw[i : i + bsize] - model.log_det / 2 - energy / 2
            torch.logaddexp(
                result[j : j + bsize],
                torch.logsumexp(log_likelihood, dim=0),
                out=result[j : j + bsize],
            )

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset",
        type=str,
        choices=[
            "gas",
            "power",
            "hepmass",
            "bsds300",
            "miniboone",
            "pinwheel",
            "2spirals",
            "checkerboard",
        ],
    )
    parser.add_argument("bandwidth", type=float)
    parser.add_argument("--num_train", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bsize", type=int, default=2000)
    args = parser.parse_args()

    print(
        "Running with dataset {} and bandwidth = {}".format(
            args.dataset, args.bandwidth
        )
    )

    if args.dataset in ["pinwheel", "2spirals", "checkerboard"]:
        rng = np.random.default_rng(args.seed)
        train_data = inf_train_gen(args.dataset, rng, batch_size=args.num_train)
        val_data = inf_train_gen(args.dataset, rng, batch_size=10_000)
        test_data = inf_train_gen(args.dataset, rng, batch_size=10_000)
    else:
        if args.dataset == "gas":
            dataset = GAS()
        elif args.dataset == "power":
            dataset = POWER()
        elif args.dataset == "hepmass":
            dataset = HEPMASS()
        elif args.dataset == "bsds300":
            dataset = BSDS300()
        elif args.dataset == "miniboone":
            dataset = MINIBOONE()

        train_data = dataset.trn.x
        val_data = dataset.val.x
        test_data = dataset.tst.x

    if 0 < args.num_train < train_data.shape[0]:
        np.random.seed(args.seed)
        inds = np.random.choice(len(train_data), args.num_train, replace=False)
        train_data = train_data[inds]

    print(train_data.shape)
    print(val_data.shape)
    print(test_data.shape)

    model = gaussian_kde(dataset=train_data.T, bw_method=args.bandwidth)
    # print(model.logpdf(test_data.T).mean())
    print(torch_logpdf(model, torch.from_numpy(val_data).cuda(), args.bsize).mean())
    print(torch_logpdf(model, torch.from_numpy(test_data).cuda(), args.bsize).mean())
