import itertools

import numpy as np


def plot_single(
    name,
    evaluator,
    degrees,
    res=50,
    scaling="normal",
    colorbar=True,
    cmap="RdBu_r",
):
    import meshzoo
    from matplotlib import pyplot as plt

    n = sum(degrees)
    r = degrees[0]

    def f(bary):
        for k, level in enumerate(evaluator(bary, scaling)):
            if k == n:
                return level[r]

    points, cells = meshzoo.disk(6, res)
    z = np.array(f(points.T), dtype=float)

    plt.tripcolor(points[:, 0], points[:, 1], cells, z, shading="flat")

    if colorbar:
        plt.colorbar()
    # Choose a diverging colormap such that the zeros are clearly distinguishable.
    plt.set_cmap(cmap)
    # Make sure the color map limits are symmetric around 0.
    clim = plt.gci().get_clim()
    mx = max(abs(clim[0]), abs(clim[1]))
    plt.clim(-mx, mx)

    # circle outline
    circle = plt.Circle((0, 0), 1.0, edgecolor="k", fill=False)
    plt.gca().add_patch(circle)

    plt.gca().set_aspect("equal")
    plt.axis("off")
    plt.title(
        f"{name} orthogonal polynomial on disk "
        f"([{degrees[0]}, {degrees[1]}], {scaling})"
    )


def savefig_tree(filename, *args, dpi=None, **kwargs):
    from matplotlib import pyplot as plt

    plot_tree(*args, **kwargs)
    plt.savefig(filename, dpi=dpi, transparent=True, bbox_inches="tight")


def show_tree(*args, **kwargs):
    from matplotlib import pyplot as plt

    plot_tree(*args, **kwargs)
    plt.show()


# Use a diverging colormap by default so the zeros are well recognizable
# https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
def plot_tree(
    name,
    evaluator,
    n,
    res=50,
    scaling="normal",
    colorbar=False,
    cmap="RdBu_r",
    clim=None,
    show_title=True,
):
    import matplotx
    import meshzoo
    from matplotlib import pyplot as plt

    plt.style.use(matplotx.styles.dufte)

    points, cells = meshzoo.disk(6, res)
    evaluator = evaluator(points.T, scaling)

    plt.set_cmap(cmap)
    plt.gca().set_aspect("equal")
    plt.axis("off")

    for k, level in enumerate(itertools.islice(evaluator, n + 1)):
        for r, z in enumerate(level):
            pts = points.copy()
            offset = [2.6 * (r - k / 2), -2.3 * k]

            pts[:, 0] += offset[0]
            pts[:, 1] += offset[1]

            plt.tripcolor(pts[:, 0], pts[:, 1], cells, z, shading="flat")
            plt.clim(clim)

            # circle outline
            circle = plt.Circle(offset, 1.0, edgecolor="k", fill=False)
            plt.gca().add_patch(circle)

    if colorbar:
        plt.colorbar()

    if show_title:
        plt.title(f"{name} orthogonal polynomials on the disk ({scaling})")
