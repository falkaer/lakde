import numpy as np
import torch

LOW = -4
HIGH = 4
SIZES = 1000
SPREAD = 0.1


def to_grid(x_min, x_max, y_min, y_max, grid_points=1000, spread=0.1, device="cpu"):
    x_spread = torch.abs(x_max - x_min) * spread
    y_spread = torch.abs(y_max - y_min) * spread
    xi = torch.linspace(x_min - x_spread, x_max + x_spread, grid_points, device=device)
    yi = torch.linspace(y_min - y_spread, y_max + y_spread, grid_points, device=device)
    return xi, yi


def support_grid(X, grid_points=1000, spread=0.1):
    assert X.shape[-1] == 2, "must be 2D"
    x, y = X[:, 0], X[:, 1]
    return to_grid(x.min(), x.max(), y.min(), y.max(), grid_points, spread, X.device)


def grid_log_density(model, X, xi, yi):
    xx, yy = torch.meshgrid(xi, yi, indexing="xy")
    Y = torch.stack([xx.flatten(), yy.flatten()], dim=-1)
    zi = model.log_pred_density(X, Y).reshape_as(xx)
    return zi


def grid_density(model, X, xi, yi):
    return grid_log_density(model, X, xi, yi).exp_()


def trapz_grid(xi, yi, zi):
    return torch.trapz(torch.trapz(zi, xi), yi)


def density_mesh(kde, X, low=None, high=None):
    if low is None and high is None:
        xi, yi = support_grid(X)
    else:
        xi, yi = to_grid(low[0], high[0], low[1], high[1], device=X.device)
    zi = grid_density(kde, X, xi, yi)
    return xi, yi, zi


def plot_colormesh(mesh, cmap="viridis", norm=None, ax=None):
    import matplotlib.pyplot as plt

    fig = None
    if ax is None:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.gca()

    xi, yi, zi = mesh
    ax.pcolormesh(xi, yi, zi, norm=norm, cmap=cmap)

    ax.set_xticks([])
    ax.set_yticks([])

    if fig is not None:
        fig.tight_layout()
        fig.show()


def plot_contour(
    mesh, levels=8, cmap="viridis", solid=True, norm=None, ax=None, **kwargs
):
    import matplotlib.pyplot as plt

    fig = None
    if ax is None:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.gca()

    xi, yi, zi = mesh

    if solid:
        cs = ax.contourf(xi, yi, zi, levels=levels, norm=norm, cmap=cmap)
        ax.contour(cs, levels=cs.levels, norm=norm, colors="k", **kwargs)
    else:
        cs = ax.contour(xi, yi, zi, levels=levels, norm=norm, colors="k", **kwargs)

    ax.set_xticks([])
    ax.set_yticks([])

    if fig is not None:
        fig.tight_layout()
        fig.show()

    return cs.levels


def plot_density_colormesh(
    kde,
    X,
    cmap="viridis",
    show_points=False,
    low=None,
    high=None,
    sizes=None,
    norm=None,
    ax=None,
):
    import matplotlib.pyplot as plt

    fig = None
    if ax is None:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.gca()

    if low is None:
        low = (LOW, LOW)
    if high is None:
        high = (HIGH, HIGH)
    if sizes is None:
        sizes = (SIZES, SIZES)

    xi = torch.linspace(low[0], high[1], sizes[0])
    yi = torch.linspace(low[1], high[1], sizes[1])

    xx, yy = np.meshgrid(xi.numpy(), yi.numpy())
    xx = torch.from_numpy(xx)
    yy = torch.from_numpy(yy)

    Y = torch.stack([xx.flatten(), yy.flatten()], dim=-1).to(X.device)
    zi = kde.log_pred_density(X, Y).exp().reshape_as(xx).cpu()
    xi, yi, zi = xi.numpy(), yi.numpy(), zi.numpy()

    ax.pcolormesh(xi, yi, zi, norm=norm, cmap=cmap)

    if show_points:
        Xn = X.cpu().numpy()
        ax.scatter(Xn[:, 0], Xn[:, 1], s=3, color="white")

    ax.set_xticks([])
    ax.set_yticks([])

    if fig is not None:
        fig.tight_layout()
        fig.show()
