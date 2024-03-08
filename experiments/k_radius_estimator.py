import argparse
import math

import numpy as np
import torch
from datasets import *
from lakde.linalg_util import lgamma

from experiments.utils import density_mesh, trapz_grid


# radius r, dimension d
def log_hypersphere_volume(r, d):
    log_unit_vol = d / 2 * math.log(math.pi) - lgamma(torch.tensor(d / 2 + 1))
    return log_unit_vol + d * torch.log(r)


class KNNDensityEstimator:
    def __init__(self, k, p=2, bsize=2000, corrected=True, const=None):
        self.k = k
        self.p = p
        self.bsize = bsize
        self.corrected = corrected
        self.const = const

    def log_pred_density(self, X, Y):
        m, d = Y.shape
        n = X.size(0)
        log_probs = torch.empty(m, device=X.device)
        coeff_k = self.k - 1 if self.corrected else self.k
        if coeff_k < 1:  # undefined
            return torch.fill_(log_probs, np.nan)
        const = -math.log(self.const) if self.const else 0
        coeff = const + math.log(coeff_k) - math.log(n)
        zero = torch.zeros((), device=X.device)
        for i in range(0, m, self.bsize):
            Yt = Y[i : i + self.bsize]
            dists = torch.cdist(Yt, X, p=self.p)
            topk = dists.topk(self.k, dim=1, largest=False, sorted=True)[0]
            r = topk[:, -1]

            # if the distance is 0 set the probability to 1
            log_probs[i : i + self.bsize] = torch.where(
                r != 0, coeff - log_hypersphere_volume(r, d), zero
            )

        return log_probs


def correct_knn_density(estimator, X):
    return trapz_grid(*density_mesh(estimator, X))


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
    parser.add_argument("k", type=int)
    parser.add_argument("--num_train", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bsize", type=int, default=500)
    args = parser.parse_args()

    # args.dataset = 'bsds300'
    # args.num_train = 500000000000000
    # args.seed = 0
    # args.k = 2

    print("Running with dataset {} and k = {}".format(args.dataset, args.k))

    if args.dataset in ["pinwheel", "2spirals", "checkerboard"]:
        rng = np.random.default_rng(args.seed)
        train_data = inf_train_gen(args.dataset, rng, batch_size=args.num_train)
        test_data = inf_train_gen(args.dataset, rng, batch_size=100_000)
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
        test_data = dataset.val.x

    if train_data.shape[0] > args.num_train:
        np.random.seed(args.seed)
        inds = np.random.choice(len(train_data), args.num_train, replace=False)
        train_data = train_data[inds]

    X = torch.from_numpy(train_data).float().cuda()
    Y = torch.from_numpy(test_data).float().cuda()

    model = KNNDensityEstimator(args.k, bsize=args.bsize, corrected=True)
    log_pdf = model.log_pred_density(X, Y).mean().item()
    print("Corrected log pdf estimate:", log_pdf)

    const = trapz_grid(*trapz_grid(*density_mesh(model, X))).item()
    print("Corrected density integrates to:", const)
    print("Numerically corrected log pdf:", log_pdf - np.log(const))

    model = KNNDensityEstimator(args.k, bsize=args.bsize, corrected=False)
    log_pdf = model.log_pred_density(X, Y).mean().item()
    print("Uncorrected log pdf estimate:", log_pdf)
