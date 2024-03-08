import numpy as np
import torch

LOW = -4
HIGH = 4
SIZES = 1000
SPREAD = 0.1


def kde_mesh(kde, X, low=None, high=None, sizes=None):
    x, y = X[:, 0], X[:, 1]
    if not (low and high):
        x_spread = torch.abs(x.max() - x.min()) * SPREAD
        y_spread = torch.abs(y.max() - y.min()) * SPREAD
        if low is None:
            low = (x.min() - x_spread, y.min() - y_spread)
        if high is None:
            high = (x.max() + x_spread, y.max() + y_spread)

    if sizes is None:
        sizes = (SIZES, SIZES)

    xi = torch.linspace(low[0], high[0], sizes[0], dtype=X.dtype, device=X.device)
    yi = torch.linspace(low[1], high[1], sizes[1], dtype=X.dtype, device=X.device)

    xx, yy = torch.meshgrid(xi, yi, indexing="xy")
    Y = torch.stack([xx.flatten(), yy.flatten()], dim=-1)
    zi = kde.pred_density(X, Y).reshape_as(xx)
    return xi, yi, zi


def trapz_grid(xi, yi, zi):
    return torch.trapz(torch.trapz(zi, xi), yi)


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


def plot_kde_colormesh(
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
