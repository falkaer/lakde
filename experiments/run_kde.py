import argparse

import numpy as np
import torch
from datasets import *
from lakde import *
from lakde.callbacks import ELBOCallback, LikelihoodCallback


def make_parser():
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
            "knn",
            "knn_inf_nu",
            "knn_fixed_tau",
            "knn_fixed_tau_inf_nu",
            "knn_shared_tau",
            "hierarchical",
            "full",
            "diag",
            "scalar",
        ],
    )

    parser.add_argument("nu_0", type=float)
    parser.add_argument("k", type=int)
    parser.add_argument("--sparsity_threshold", type=float, default=5e-4)

    parser.add_argument("--num_train", type=int, default=-1)
    parser.add_argument("--no_elbo", default=False, action="store_true")
    parser.add_argument("--shuffle", default=False, action="store_true")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument(
        "--validate_on", type=str, choices=["iteration", "epoch"], default="epoch"
    )
    parser.add_argument("--validate_every", type=int, default=5)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bsize", type=int, default=5000)
    parser.add_argument("--verbose", type=bool, default=True)
    parser.add_argument("--logs", type=bool, default=True)

    parser.add_argument("--plot", default=False, action="store_true")
    parser.add_argument("--plot_out", type=str, default=None)
    parser.add_argument("--verify_density", default=False, action="store_true")

    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--save_best", default=False, action="store_true")
    parser.add_argument("--save_to", type=str, default=None)
    parser.add_argument("--rtol", default=None, action="store_true")
    return parser


def run_kde(args):
    if args.dataset in ["pinwheel", "2spirals", "checkerboard"]:
        rng = np.random.default_rng(args.seed)
        train_data = inf_train_gen(args.dataset, rng, batch_size=args.num_train)
        val_data = inf_train_gen(args.dataset, rng, batch_size=50_000)
        test_data = inf_train_gen(args.dataset, rng, batch_size=50_000)
        dataset_name = args.dataset
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
        dataset_name = dataset.__class__.__name__

    if train_data.shape[0] > args.num_train > 0:
        np.random.seed(args.seed)
        inds = np.random.choice(len(train_data), args.num_train, replace=False)
        train_data = train_data[inds]

    train_data = torch.as_tensor(train_data).float().cuda().contiguous()
    val_data = torch.as_tensor(val_data).float().cuda().contiguous()
    test_data = torch.as_tensor(test_data).float().cuda().contiguous()

    print("Training data: {}".format(train_data.shape))
    print("Validation data: {}".format(val_data.shape))
    print("Test data: {}".format(test_data.shape))

    if args.shuffle:
        gen = torch.Generator().manual_seed(args.seed)
        perm = torch.randperm(train_data.size(0), generator=gen)
        train_data = train_data[perm]

    callbacks = []

    if not args.no_elbo:
        callbacks.append(
            ELBOCallback(report_on="iteration", report_every=1, verbose=args.verbose)
        )

    likelihood_kwargs = {
        "report_on": args.validate_on,
        "report_every": args.validate_every,
        "verbose": args.verbose,
    }
    callbacks.append(
        LikelihoodCallback(
            val_data,
            eval_label="val",
            save_best=args.save_best,
            save_to=args.save_to,
            rtol=args.rtol,
            **likelihood_kwargs,
        )
    )
    callbacks.append(
        LikelihoodCallback(test_data, eval_label="test", **likelihood_kwargs)
    )

    if args.model == "hierarchical":
        model = LocalHierarchicalKDE(
            nu_0=args.nu_0,
            k_or_knn_graph=args.k,
            block_size=args.bsize,
            verbose=args.verbose,
            logs=args.logs,
        )
    elif args.model == "knn":
        model = LocalKNearestKDE(
            nu_0=args.nu_0,
            k_or_knn_graph=args.k,
            block_size=args.bsize,
            verbose=args.verbose,
            logs=args.logs,
        )
    elif args.model == "knn_inf_nu":
        model = LocalKNearestKDEInfNu(
            nu_0=args.nu_0,
            k_or_knn_graph=args.k,
            block_size=args.bsize,
            verbose=args.verbose,
            logs=args.logs,
        )
    elif args.model == "knn_fixed_tau":
        model = LocalKNearestKDEFixedTau(
            nu_0=args.nu_0,
            k_or_knn_graph=args.k,
            block_size=args.bsize,
            verbose=args.verbose,
            logs=args.logs,
        )

    elif args.model == "knn_fixed_tau_inf_nu":
        model = LocalKNearestKDEPriorOnly(
            nu_0=args.nu_0,
            k_or_knn_graph=args.k,
            block_size=args.bsize,
            verbose=args.verbose,
            logs=args.logs,
        )

    elif args.model == "knn_shared_tau":
        model = LocalKNearestKDESharedTau(
            nu_0=args.nu_0,
            k_or_knn_graph=args.k,
            block_size=args.bsize,
            verbose=args.verbose,
            logs=args.logs,
        )
    elif args.model == "full":
        model = SharedFullKDE(
            block_size=args.bsize, verbose=args.verbose, logs=args.logs
        )
    elif args.model == "diag":
        model = SharedDiagonalizedKDE(
            block_size=args.bsize, verbose=args.verbose, logs=args.logs
        )
    elif args.model == "scalar":
        model = SharedScalarKDE(
            block_size=args.bsize, verbose=args.verbose, logs=args.logs
        )

    print(
        "Running {} on {} with nu_0={} and k={}".format(
            model.__class__.__name__, dataset_name, args.nu_0, args.k
        )
    )

    if args.resume_from is not None:
        print("Resuming from checkpoint {}, loading parameters...")
        d = torch.load(args.resume_from)
        model.load_state_dict(d)

    model.fit(
        train_data,
        iterations=args.iterations,
        callbacks=callbacks,
        sparsity_threshold=args.sparsity_threshold,
        dataset_label=dataset_name,
    )

    if args.verify_density:
        assert args.dataset in ["pinwheel", "2spirals", "checkerboard"]
        from utils import trapz_grid, density_mesh

        numerical_const = trapz_grid(*density_mesh(model, train_data))
        print(
            "Numerically verified density constant: {}".format(numerical_const.item())
        )

    if args.plot:
        assert args.dataset in ["pinwheel", "2spirals", "checkerboard"]
        import matplotlib.pyplot as plt
        from experiments.utils import plot_density_colormesh

        plt.style.use(["science", "grid"])
        norm = plt.Normalize()

        fig = plt.figure(figsize=(10, 10))
        ax = fig.gca()
        plot_density_colormesh(model, train_data, norm=norm, ax=ax)

        if args.plot_out is not None:
            extent = (
                plt.gca()
                .get_window_extent()
                .transformed(fig.dpi_scale_trans.inverted())
            )
            plt.savefig(args.plot_out, bbox_inches=extent)
        else:
            fig.show()


if __name__ == "__main__":
    parser = make_parser()
    run_kde(parser.parse_args())
