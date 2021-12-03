import itertools

import numpy as np

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
    import matplotx
    import meshzoo
    from matplotlib import pyplot as plt

    plt.style.use(matplotx.styles.dufte)

    points, cells = meshzoo.rectangle(-1.0, 1.0, -1.0, 1.0, res, res)
    evaluator = Eval(points.T, *args, **kwargs)

    plt.set_cmap(cmap)
    plt.gca().set_aspect("equal")
    plt.axis("off")

    for k, values in enumerate(itertools.islice(evaluator, n + 1)):
        for r, z in enumerate(values):
            offset = [2.8 * (r - k / 2), -2.6 * k]
            pts = points + offset

            plt.tripcolor(pts[:, 0], pts[:, 1], cells, z, shading="flat")
            plt.clim(clim)

            # rectangle outlines
            corners = np.array([[-1, 1, 1, -1, -1], [-1, -1, 1, 1, -1]])
            corners = (corners.T + offset).T
            plt.plot(corners[0], corners[1], "-k")

    if colorbar:
        plt.colorbar()


def write_tree_3d(filename, n, *args, res=20, **kwargs):
    import meshio
    import meshzoo

    points, cells = meshzoo.cube(-1, 1, -1, 1, -1, 1, res, res, res)

    evaluator = Eval(points.T, *args, return_degrees=True, **kwargs)
    meshes = []

    corners = np.array(
        [[np.cos(k * 2 * np.pi / 3), np.sin(k * 2 * np.pi / 3)] for k in range(3)]
    )

    for L, (values, degrees) in enumerate(itertools.islice(evaluator, n)):
        for vals, degs in zip(values, degrees):
            offset = sum([corner * d for corner, d in zip(corners, degs)])
            pts = points.copy()
            pts[:, 0] += 2.0 * offset[0]
            pts[:, 1] += 2.0 * offset[1]
            pts[:, 2] -= 3.0 * L
            meshes.append(meshio.Mesh(pts, {"tetra": cells}, point_data={"f": vals}))

    # merge meshes
    points = np.concatenate([mesh.points for mesh in meshes])
    f_vals = np.concatenate([mesh.point_data["f"] for mesh in meshes])
    #
    cells = []
    k = 0
    for mesh in meshes:
        cells.append(mesh.cells[0].data + k)
        k += mesh.points.shape[0]
    cells = np.concatenate(cells)

    meshio.write_points_cells(
        filename, points, {"tetra": cells}, point_data={"f": f_vals}
    )
