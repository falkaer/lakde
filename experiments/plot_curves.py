import matplotlib.pyplot as plt
import numpy as np
import scienceplots
import torch
from lakde import *
from matplotlib.ticker import AutoMinorLocator, FixedLocator

from experiments.run_curves import all_num_train

plt.style.use(["science", "grid"])

datasets = ["pinwheel", "2spirals", "checkerboard"]


def style_axis(ax, yticks):
    ax.set_yticks(yticks)
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.set_xscale("log")
    ax.set_xticks(
        [10**2, 10**3, 10**4, 10**5],
        [r"$10^2$", r"$10^3$", r"$10^4$", r"$10^5$"],
    )
    ax.xaxis.set_minor_locator(FixedLocator(all_num_train))
    ax.set_xlim(50, 10**5)


def plot_curve(ax, means, stds=None, linestyle="solid", **kwargs):
    if stds is not None:
        ax.errorbar(all_num_train, means, yerr=stds, marker="o", markersize=2, **kwargs)
    else:
        ax.plot(all_num_train, means, "o", linestyle=linestyle, markersize=2, **kwargs)


def plot_runs():
    f, allaxes = plt.subplots(3, 1, figsize=(2, 5), dpi=200)

    for ax, dataset in zip(allaxes, datasets):
        k_radius_curves = torch.load(
            "curves/k_radius/{}_curves.pt".format(dataset), map_location="cpu"
        )
        plot_curve(
            ax,
            [
                np.mean(k_radius_curves[str(n) + "_uncorrected_ll"])
                for n in all_num_train
            ],
            linestyle="dotted",
            alpha=0.5,
            label=r"KNN $(K)$",
        )
        plot_curve(
            ax,
            [np.mean(k_radius_curves[str(n) + "_corrected_ll"]) for n in all_num_train],
            linestyle="dotted",
            alpha=0.5,
            label=r"KNN $(K-1)$",
        )
        plot_curve(
            ax,
            [np.mean(k_radius_curves[str(n)]) for n in all_num_train],
            [np.std(k_radius_curves[str(n)]) for n in all_num_train],
            label=r"KNN",
        )

        full_kde_curves = torch.load(
            "curves/full/{}_curves.pt".format(dataset), map_location="cpu"
        )
        plot_curve(
            ax,
            [np.mean(full_kde_curves[str(n)]) for n in all_num_train],
            [np.std(full_kde_curves[str(n)]) for n in all_num_train],
            label=r"VB-Full-KDE",
        )

        knn_kde_curves = torch.load(
            "curves/knn/{}_curves.pt".format(dataset), map_location="cpu"
        )
        plot_curve(
            ax,
            [np.mean(knn_kde_curves[str(n)]) for n in all_num_train],
            [np.std(knn_kde_curves[str(n)]) for n in all_num_train],
            label=r"VB-KNN-KDE",
        )
        ax.set_title(dataset)

    style_axis(allaxes[0], [-2.2, -2.5, -2.8])
    style_axis(allaxes[1], [-2.5, -3.0, -3.5])
    style_axis(allaxes[2], [-3.5, -3.8, -4.1])
    plt.subplots_adjust(hspace=0.4)
    allaxes[-1].legend(
        loc="lower center",
        ncol=2,
        columnspacing=0.9,
        fontsize=7,
        bbox_to_anchor=[0.4, -0.7],
    )


if __name__ == "__main__":
    plot_runs()
    plt.savefig("plots/curves.png")
