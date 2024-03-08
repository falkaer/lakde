import argparse
import os
import os.path as osp

import ax
import numpy as np
import torch
from ax.service.utils.best_point import get_best_parameters
from ax.storage.json_store.load import load_experiment
from datasets import *
from lakde import *

from experiments.k_radius_estimator import KNNDensityEstimator
from experiments.run_utils import run_all


def parse_args():
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
    parser.add_argument(
        "model",
        type=str,
        choices=[
            "hierarchical",
            "knn",
            "full",
            "diag",
            "scalar",
            "k_radius",
        ],
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bsize", type=int, default=10_000)
    parser.add_argument("--sparsity_threshold", type=float, default=5e-4)
    return parser.parse_args()


SAVE_ROOT = "hparams"
all_num_train = [100, 250, 500, 1000, 2500, 5000, 10_000, 25_000, 50_000]
num_subsets = 5


def get_closest_hparams(model, dataset, num_train):
    # find largest available hparam run with N <= num_train
    ind = next(i for i, x in reversed(list(enumerate(all_num_train))) if x <= num_train)
    while ind >= 0:
        num_train = all_num_train[ind]
        path = osp.join(
            SAVE_ROOT,
            model,
            dataset,
            "{}_{}_{}_experiment.json".format(model, dataset, num_train),
        )
        if osp.exists(path):
            experiment = load_experiment(path)
            return get_best_parameters(experiment, ax.Models)[0]
        else:
            ind -= 1  # try next if it isn't found


if __name__ == "__main__":
    args = parse_args()

    print(
        "Running curves for {} on {} at {} sparsity levels".format(
            args.model, args.dataset, args.sparsity_threshold
        )
    )

    def model_supplier(num_train):
        print("Instantiating a {} model...".format(args.model))
        if args.model in ["knn", "hierarchical", "k_radius"]:
            hparams = get_closest_hparams(args.model, args.dataset, num_train)

        if args.model == "knn":
            print("Found hparams: nu_0={}, k={}".format(hparams["nu_0"], hparams["k"]))
            model = LocalKNearestKDE(
                nu_0=hparams["nu_0"],
                k_or_knn_graph=hparams["k"],
                block_size=args.bsize,
                verbose=True,
                logs=False,
            )
        elif args.model == "hierarchical":
            print("Found hparams: nu_0={}".format(hparams["nu_0"]))
            model = LocalHierarchicalKDE(
                nu_0=hparams["nu_0"],
                k_or_knn_graph=min(num_train - 1, 500),
                block_size=args.bsize,
                verbose=True,
                logs=False,
            )
        elif args.model == "full":
            model = SharedFullKDE(block_size=args.bsize, verbose=True, logs=False)
        elif args.model == "diag":
            model = SharedDiagonalizedKDE(
                block_size=args.bsize, verbose=True, logs=False
            )
        elif args.model == "scalar":
            model = SharedScalarKDE(block_size=args.bsize, verbose=True, logs=False)
        elif args.model == "k_radius":
            model = KNNDensityEstimator(k=hparams["k"], bsize=args.bsize)
        return model

    rng = np.random.default_rng(args.seed)
    if args.dataset in ["pinwheel", "2spirals", "checkerboard"]:

        def train_data_supplier(size):
            return inf_train_gen(args.dataset, rng, batch_size=size)

        val_data = inf_train_gen(args.dataset, rng, batch_size=50_000)
        test_data = inf_train_gen(args.dataset, rng, batch_size=1_000_000)
        dataset_name = args.dataset
        ll_rtol = 1e-4
        validate_every = 5

    else:
        if args.dataset == "gas":
            dataset = GAS()
            ll_rtol = 1e-4
        elif args.dataset == "power":
            dataset = POWER()
            ll_rtol = 1e-4
        elif args.dataset == "hepmass":
            dataset = HEPMASS()
            ll_rtol = 4e-3
        elif args.dataset == "bsds300":
            dataset = BSDS300()
            ll_rtol = 1e-2
        elif args.dataset == "miniboone":
            dataset = MINIBOONE()
            ll_rtol = 2e-3

        validate_every = 1
        train_data = dataset.trn.x

        def train_data_supplier(size):
            sample_size = min(len(train_data), size)
            inds = rng.choice(len(train_data), sample_size, replace=False)
            return train_data[inds]

        val_data = dataset.val.x
        test_data = dataset.tst.x
        dataset_name = dataset.__class__.__name__

    metrics = run_all(
        train_data_supplier=train_data_supplier,
        val_data=val_data,
        test_data=test_data,
        model_supplier=model_supplier,
        sample_sizes=all_num_train,
        num_subsets=num_subsets,
        threshold=args.sparsity_threshold,
        max_iterations=100_000,  # stopping only when relative tolerance is reached
        ll_rtol=ll_rtol,
        validate_every=validate_every,
    )

    path = "curves ({})/{}/{}_curves.pt".format(
        args.sparsity_threshold, args.model, args.dataset
    )
    os.makedirs(osp.dirname(path), exist_ok=True)
    torch.save(metrics, path)
