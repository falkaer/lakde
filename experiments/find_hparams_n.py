import argparse
import os
import os.path as osp

from ax.storage.json_store.save import save_experiment
from datasets import *

from experiments.find_hparams import (
    add_arguments,
    all_total_trials,
    param_dict,
    run_experiment,
)

SAVE_ROOT = "hparams"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("num_train", type=int, default=50_000)
    args = add_arguments(parser).parse_args()

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
    total_trials = all_total_trials[-1]
    nu_range = params["nu_range"]
    k_range = params["k_range"]

    path = osp.join(
        SAVE_ROOT,
        args.model,
        args.dataset,
        "{}_{}_{}_experiment_{}.json".format(
            args.model, args.dataset, args.num_train, args.seed
        ),
    )
    print(
        "Running experiment for N={} on {} ({} tolerance, {} trials)".format(
            args.num_train, args.dataset, ll_rtol, total_trials
        )
    )

    best_parameters, experiment, _ = run_experiment(
        dataset,
        args.model,
        args.bsize,
        args.num_train,
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
