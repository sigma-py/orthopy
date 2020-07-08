import itertools

import numpy

from ..c1.jacobi import plot as plot_jacobi
from .main import Eval


def show_tree_1d(*args, **kwargs):
    from matplotlib import pyplot as plt

    plot_tree_1d(*args, **kwargs)
    plt.show()
    plt.close()


def savefig_tree_1d(filename, *args, **kwargs):
    from matplotlib import pyplot as plt

    plot_tree_1d(*args, **kwargs)
    plt.savefig(filename, transparent=True, bbox_inches="tight")
    plt.close()


def plot_tree_1d(n, *args, **kwargs):
    plot_jacobi(n, *args, **kwargs)


def show_tree_2d(*args, **kwargs):
    from matplotlib import pyplot as plt

    plot_tree_2d(*args, **kwargs)
    plt.show()
    plt.close()


def savefig_tree_2d(filename, *args, **kwargs):
    from matplotlib import pyplot as plt

    plot_tree_2d(*args, **kwargs)
    plt.savefig(filename, transparent=True, bbox_inches="tight")
    plt.close()


def plot_tree_2d(n, *args, res=100, colorbar=True, cmap="RdBu_r", clim=None, **kwargs):
    import dufte
    import meshzoo
    from matplotlib import pyplot as plt

    plt.style.use(dufte.style)

    points, cells = meshzoo.rectangle(-1.0, 1.0, -1.0, 1.0, res, res)
    evaluator = Eval(points.T, *args, **kwargs)

    plt.set_cmap(cmap)
    plt.gca().set_aspect("equal")
    plt.axis("off")

    for k, level in enumerate(itertools.islice(evaluator, n + 1)):
        for r, z in enumerate(level):
            offset = [2.8 * (r - k / 2), -2.6 * k]
            pts = points + offset

            plt.tripcolor(pts[:, 0], pts[:, 1], cells, z, shading="flat")
            plt.clim(clim)

            # rectangle outlines
            corners = numpy.array([[-1, 1, 1, -1, -1], [-1, -1, 1, 1, -1]])
            corners = (corners.T + offset).T
            plt.plot(corners[0], corners[1], "-k")

    if colorbar:
        plt.colorbar()
