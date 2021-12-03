import itertools

import numpy as np

from .main import Eval


def savefig_single(filename, *args, **kwargs):
    from matplotlib import pyplot as plt

    plot_single(*args, **kwargs)
    plt.savefig(filename, transparent=True, bbox_inches="tight")


def show_single(*args, **kwargs):
    from matplotlib import pyplot as plt

    plot_single(*args, **kwargs)
    plt.show()


def plot_single(
    degrees, res=100, scaling="normal", colorbar=True, cmap="RdBu_r", corners=None
):
    import meshzoo
    from matplotlib import pyplot as plt

    n = sum(degrees)
    r = degrees[0]

    def f(bary):
        for k, level in enumerate(Eval(bary, scaling)):
            if k == n:
                return level[r]

    if corners is None:
        alpha = np.pi * np.array([7.0 / 6.0, 11.0 / 6.0, 3.0 / 6.0])
        corners = np.array([np.cos(alpha), np.sin(alpha)])

    bary, cells = meshzoo.triangle(res)
    x, y = np.dot(corners, bary)
    z = np.array(f(bary), dtype=float)

    plt.tripcolor(x, y, cells, z, shading="flat")

    if colorbar:
        plt.colorbar()
    # Choose a diverging colormap such that the zeros are clearly distinguishable.
    plt.set_cmap(cmap)
    # Make sure the color map limits are symmetric around 0.
    clim = plt.gci().get_clim()
    mx = max(abs(clim[0]), abs(clim[1]))
    plt.clim(-mx, mx)

    # triangle outlines
    X = np.column_stack([corners, corners[:, 0]])
    plt.plot(X[0], X[1], "-k")

    plt.gca().set_aspect("equal")
    plt.axis("off")
    plt.title(
        f"Orthogonal polynomial on triangle ([{degrees[0]}, {degrees[1]}], {scaling})"
    )


def savefig_tree(filename, *args, **kwargs):
    from matplotlib import pyplot as plt

    plot_tree(*args, **kwargs)
    plt.savefig(filename, transparent=True, bbox_inches="tight")


def show_tree(*args, **kwargs):
    from matplotlib import pyplot as plt

    plot_tree(*args, **kwargs)
    plt.show()


# Use a diverging colormap by default so the zeros are well recognizable
# https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
def plot_tree(n, res=100, scaling="normal", colorbar=False, cmap="RdBu_r", clim=None):
    import matplotx
    import meshzoo
    from matplotlib import pyplot as plt

    plt.style.use(matplotx.styles.dufte)

    bary, cells = meshzoo.triangle(res)
    evaluator = Eval(bary, scaling)

    plt.set_cmap(cmap)
    plt.gca().set_aspect("equal")
    plt.axis("off")

    for k, level in enumerate(itertools.islice(evaluator, n + 1)):
        for r, z in enumerate(level):
            alpha = np.pi * np.array([7.0 / 6.0, 11.0 / 6.0, 3.0 / 6.0])
            corners = np.array([np.cos(alpha), np.sin(alpha)])
            corners[0] += 2.1 * (r - k / 2)
            corners[1] -= 1.9 * k
            x, y = np.dot(corners, bary)

            plt.tripcolor(x, y, cells, z, shading="flat")
            plt.clim(clim)

            # triangle outlines
            X = np.column_stack([corners, corners[:, 0]])
            plt.plot(X[0], X[1], "-k")

    if colorbar:
        plt.colorbar()
