import argparse

import ax
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from ax.modelbridge.factory import get_GPMES
from ax.plot.contour import _get_contour_predictions
from ax.plot.helper import get_range_parameter
from ax.service.utils.best_point import get_best_parameters
from ax.storage.json_store.load import load_experiment

matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams.update({"font.size": 24})


def plot_max_point(x, y, ax):
    ax.scatter(x, y, c="gold", edgecolors="black", marker="*", s=400, zorder=3)


def plot_acq_points(x_points, y_points, ax):
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    ax.scatter(x_points, y_points, c="red", marker="x", s=20, zorder=1)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


def plot_contour(xi, yi, zi, levels, norm, cmap, ax):
    colormesh = ax.pcolormesh(
        xi, yi, zi, shading="gouraud", cmap=cmap, norm=norm, antialiased=True
    )
    contours = ax.contour(
        xi,
        yi,
        zi,
        levels,
        colors="black",
        linestyles="solid",
        norm=norm,
        zorder=2,
        antialiased=True,
    )

    ax.clabel(contours, inline=True, fontsize=20, inline_spacing=16)
    plt.colorbar(colormesh, ax=ax, boundaries=contours.levels, drawedges=True)


def plot_posterior_contour(
    model, experiment, axes, f_levels=None, sd_levels=None, cmap="viridis"
):
    _, f_plt, sd_plt, xi, yi, scales = _get_contour_predictions(
        model, "nu_0", "k", "log_likelihood", None, 200
    )
    hparams = get_best_parameters(experiment, ax.Models)[0]
    trials = experiment.trials
    max_x = hparams["nu_0"]
    max_y = hparams["k"]

    x_points = [t.arm.parameters["nu_0"] for t in trials.values()]
    y_points = [t.arm.parameters["k"] for t in trials.values()]

    f_plt = np.array(f_plt).reshape(len(xi), len(yi))
    sd_plt = np.array(sd_plt).reshape(len(xi), len(yi))

    if not isinstance(f_levels, int):
        norm = plt.Normalize(f_levels[0], f_levels[-1], clip=True)
    else:
        norm = None

    # plot function values
    plot_contour(xi, yi, f_plt, f_levels, norm, cmap, axes[0])
    plot_acq_points(x_points, y_points, axes[0])
    plot_max_point(max_x, max_y, axes[0])

    # plot standard deviation
    plot_contour(xi, yi, sd_plt, sd_levels, None, "Blues", axes[1])
    plot_acq_points(x_points, y_points, axes[1])

    axes[0].set_xlabel(r"$\nu$", fontsize=28)
    axes[0].set_ylabel(r"$K$", fontsize=28)
    axes[1].set_xlabel(r"$\nu$", fontsize=28)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_path", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    experiment = load_experiment(args.experiment_path)
    model = get_GPMES(experiment, experiment.fetch_data())
    hparams = get_best_parameters(experiment, ax.Models)[0]

    for k, v in hparams.items():
        param = get_range_parameter(model, k)
        print(
            "Hyperparameter {} in range [{}, {}], optimal value: {}".format(
                k, param.lower, param.upper, v
            )
        )

    print("Plotting {} experiment".format(args.experiment_path))

    f_levels = 8
    sd_levels = 8

    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(20, 8), dpi=100)
    plot_posterior_contour(model, experiment, axes, f_levels, sd_levels)

    fig.tight_layout()
    fig.show()
