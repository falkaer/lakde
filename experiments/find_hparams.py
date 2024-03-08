import argparse
import os
import os.path as osp

import numpy as np
import torch
from ax import Models
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.service.managed_loop import optimize
from ax.storage.json_store.save import save_experiment
from datasets import *
from lakde import (
    LocalHierarchicalKDE,
    LocalKNearestKDE,
    LocalKNearestKDEFixedTau,
    LocalKNearestKDEInfNu,
    LocalKNearestKDESharedTau,
)

from experiments.k_radius_estimator import KNNDensityEstimator
from experiments.run_utils import get_subset, run_model


def run_experiment(
    data,
    model_type,
    block_size,
    num_train,
    nu_range,
    k_range,
    ll_rtol,
    total_trials,
    threshold,
    validate_every,
    seed,
):
    torch.manual_seed(seed)
    root_seq = np.random.SeedSequence(seed)

    def eval_func(param):
        nu_0 = param.get("nu_0")
        k = param.get("k")
        if model_type == "knn":
            # k = param.get('k')
            print("Running on nu_0={}, k={}... ".format(nu_0, k), end="")
            model = LocalKNearestKDE(
                nu_0=nu_0,
                k_or_knn_graph=k,
                block_size=block_size,
                logs=False,
                verbose=True,
            )
        elif model_type == "knn_inf_nu":
            print("Running on nu_0={}, k={}... ".format(nu_0, k), end="")
            model = LocalKNearestKDEInfNu(
                nu_0=nu_0,
                k_or_knn_graph=k,
                block_size=block_size,
                logs=False,
                verbose=True,
            )
        elif model_type == "knn_fixed_tau":
            print("Running on nu_0={}, k={}... ".format(nu_0, k), end="")
            model = LocalKNearestKDEFixedTau(
                nu_0=nu_0,
                k_or_knn_graph=k,
                block_size=block_size,
                logs=False,
                verbose=True,
            )
        elif model_type == "knn_shared_tau":
            print("Running on nu_0={}, k={}... ".format(nu_0, k), end="")
            model = LocalKNearestKDESharedTau(
                nu_0=nu_0,
                k_or_knn_graph=k,
                block_size=block_size,
                logs=False,
                verbose=True,
            )
        elif model_type == "hierarchical":
            print("Running on nu_0={}, k={}... ".format(nu_0, k), end="")
            model = LocalHierarchicalKDE(
                nu_0=nu_0,
                k_or_knn_graph=k,
                block_size=block_size,
                logs=False,
                verbose=True,
            )
        elif model_type == "k_radius":
            print("Running on k={}... ".format(k), end="")
            model = KNNDensityEstimator(k=k, bsize=block_size)

        seq = root_seq.spawn(1)[0]

        if isinstance(data, str):
            rng = np.random.default_rng(seq)
            train_data = inf_train_gen(data, rng=rng, batch_size=num_train)
            val_data = inf_train_gen(data, rng=rng, batch_size=50_000)
        else:
            train_data = get_subset(data.trn.x, num_train, seq)
            val_data = data.val.x

        test_ll, meta = run_model(
            model,
            train_data,
            val_data,
            val_data,  # use val data for final score when searching for hparams
            threshold=threshold,
            ll_rtol=ll_rtol,
            validate_every=validate_every,
        )
        print("Finished, got ll={}".format(test_ll))
        return test_ll

    # clamp so k < N
    k_range = [k_range[0], min(num_train - 1, k_range[1])]

    parameters = [
        {"name": "nu_0", "type": "range", "value_type": "float", "bounds": nu_range},
        {"name": "k", "type": "range", "value_type": "int", "bounds": k_range},
    ]

    # use a Sobol sequence for the first 5 points, then MES
    gs = GenerationStrategy(
        steps=[
            GenerationStep(
                model=Models.SOBOL,
                num_trials=5,
                min_trials_observed=5,
            ),
            GenerationStep(
                model=Models.GPMES,
                num_trials=-1,
            ),
        ]
    )

    best_parameters, values, experiment, model = optimize(
        parameters=parameters,
        evaluation_function=eval_func,
        generation_strategy=gs,
        objective_name="log_likelihood",
        total_trials=total_trials,
        random_seed=seed,
    )
    return best_parameters, experiment, model


def add_arguments(parser):
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
            "knn_inf_nu",
            "knn_fixed_tau",
            "knn_shared_tau",
            "k_radius",
        ],
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bsize", type=int, default=20_000)
    parser.add_argument("--sparsity_threshold", type=float, default=5e-4)
    parser.add_argument("--validate_every", type=int, default=1)
    return parser


SAVE_ROOT = "hparams"

param_dict = {
    "gas": {
        "ll_rtol": 1e-4,
        "nu_range": [8 + 0.1, 200],
        "k_range": [8 + 1, 100],
    },
    "power": {"ll_rtol": 1e-4, "nu_range": [6 + 0.1, 300], "k_range": [6 + 1, 500]},
    "hepmass": {"ll_rtol": 4e-3, "nu_range": [21 + 0.1, 300], "k_range": [21 + 1, 800]},
    "bsds300": {
        "ll_rtol": 1e-2,
        "nu_range": [63 + 0.1, 200],
        "k_range": [63 + 1, 2000],
    },
    "miniboone": {
        "ll_rtol": 2e-3,
        "nu_range": [43 + 0.1, 300],
        "k_range": [43 + 1, 800],
    },
    "pinwheel": {"ll_rtol": 1e-4, "nu_range": [2 + 0.1, 300], "k_range": [2 + 1, 800]},
    "2spirals": {"ll_rtol": 1e-4, "nu_range": [2 + 0.1, 300], "k_range": [2 + 1, 800]},
    "checkerboard": {
        "ll_rtol": 1e-4,
        "nu_range": [2 + 0.1, 300],
        "k_range": [2 + 1, 800],
    },
}

all_num_train = [100, 250, 500, 1000, 2500, 5000, 10_000, 50_000, 100_000]
all_total_trials = [75, 75, 75, 75, 75, 50, 50, 50, 50]
verbose = False

if __name__ == "__main__":
    args = add_arguments(argparse.ArgumentParser()).parse_args()

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
    elif args.dataset == "pinwheel":
        dataset = args.dataset
    elif args.dataset == "2spirals":
        dataset = args.dataset
    elif args.dataset == "checkerboard":
        dataset = args.dataset

    params = param_dict[args.dataset]
    ll_rtol = params["ll_rtol"]
    nu_range = params["nu_range"]

    if args.model == "k_radius":
        k_range = [2 + 1, 100]  # nothing good past 100
    else:
        k_range = params["k_range"]

    for num_train, total_trials in zip(all_num_train, all_total_trials):
        path = osp.join(
            SAVE_ROOT,
            args.model,
            args.dataset,
            "{}_{}_{}_experiment.json".format(args.model, args.dataset, num_train),
        )
        if osp.exists(path):
            print(
                "Experiment for N={} on {} already exists, skipping".format(
                    num_train, args.dataset
                )
            )
        else:
            print(
                "Running experiment for N={} on {} ({} tolerance, {} trials)".format(
                    num_train, args.dataset, ll_rtol, total_trials
                )
            )

            best_parameters, experiment, _ = run_experiment(
                dataset,
                args.model,
                args.bsize,
                num_train,
                nu_range,
                k_range,
                ll_rtol,
                total_trials,
                args.sparsity_threshold,
                args.validate_every,
                args.seed,
            )

            os.makedirs(osp.dirname(path), exist_ok=True)
            save_experiment(experiment, path)
