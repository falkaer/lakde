import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import inf_train_gen
from lakde import *
from lakde.callbacks import LikelihoodCallback

from experiments.k_radius_estimator import KNNDensityEstimator, correct_knn_density
from experiments.utils import plot_density_colormesh

plt.style.use(["science", "grid"])

if __name__ == "__main__":
    datasets = ["pinwheel", "2spirals", "checkerboard"]
    norms = {"pinwheel": 0.25, "2spirals": 0.15, "checkerboard": 0.06}
    knn_k = {"pinwheel": 10, "2spirals": 10, "checkerboard": 10}
    knnkde_k = {"pinwheel": 38, "2spirals": 22, "checkerboard": 29}

    f, allaxes = plt.subplots(3, 3, figsize=(6, 6), dpi=200)
    first = True

    for dataset, axes in zip(datasets, allaxes):
        rng = np.random.default_rng(0)
        train_data = inf_train_gen(dataset, rng, batch_size=1000)
        val_data = inf_train_gen(dataset, rng, batch_size=10_000)

        X = torch.from_numpy(train_data).float().cuda()

        knn = KNNDensityEstimator(k=knn_k[dataset])
        fullkde = SharedFullKDE(verbose=False)
        knnkde = LocalKNearestKDE(k_or_knn_graph=knnkde_k[dataset], verbose=False)

        for kde in [fullkde, knnkde]:
            early_stop = LikelihoodCallback(
                torch.from_numpy(val_data).float().cuda(),
                report_on="iteration",
                report_every=1,
                rtol=0,  # stop on overfit
                verbose=True,
            )
            kde.fit(X, iterations=200, callbacks=[early_stop])

        norm = plt.Normalize(vmin=0, vmax=norms[dataset], clip=True)

        knn.const = correct_knn_density(knn, X)
        plot_density_colormesh(knn, X, norm=norm, ax=axes[0])
        axes[0].set_title("KNN" if first else None)

        plot_density_colormesh(fullkde, X, norm=norm, ax=axes[1])
        axes[1].set_title("VB-Full-KDE" if first else None)

        plot_density_colormesh(knnkde, X, norm=norm, ax=axes[2])
        axes[2].set_title("VB-KNN-KDE" if first else None)

        first = False
        print(norm.vmin, norm.vmax)

    plt.tight_layout()
    plt.savefig("plots/toy_comparison_1k.png")
