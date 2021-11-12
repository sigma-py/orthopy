import itertools

import numpy as np

from .main import EvalCartesian


def write_single(filename, n, r, scaling, res=20):
    """This function creates a sphere mesh with "srgb1" values. Can be views in ParaView
    by disabling "Map Scalars".
    """
    import cplot
    import meshio
    import meshzoo

    points, cells = meshzoo.icosa_sphere(res)

    evaluator = EvalCartesian(points.T, scaling, complex_valued=True)
    vals = next(itertools.islice(evaluator, n, None))[r]

    srgb1_vals = cplot.get_srgb1(vals, colorspace="cam16")

    meshio.write_points_cells(
        filename, points, {"triangle": cells}, point_data={"srgb1": srgb1_vals}
    )


def write_tree(filename, n, scaling, res=20):
    import cplot
    import meshio
    import meshzoo

    points, cells = meshzoo.icosa_sphere(res)

    evaluator = EvalCartesian(points.T, scaling, complex_valued=True)
    meshes = []
    for L, level in enumerate(itertools.islice(evaluator, n)):
        for k, vals in enumerate(level):
            srgb1_vals = cplot.get_srgb1(vals, colorspace="cam16")
            # # exaggerate colors a bit
            # srgb1_vals *= 2.5
            # srgb1_vals[srgb1_vals > 1] = 1
            # srgb1_vals[srgb1_vals < 0] = 0

            pts = points.copy()

            pts[:, 0] += 2.2 * (k - L)
            pts[:, 2] -= 2.7 * L

            meshes.append(
                meshio.Mesh(pts, {"triangle": cells}, point_data={"srgb1": srgb1_vals})
            )

    # merge meshes
    points = np.concatenate([mesh.points for mesh in meshes])
    srgb1_vals = np.concatenate([mesh.point_data["srgb1"] for mesh in meshes])
    #
    cells = []
    k = 0
    for mesh in meshes:
        cells.append(mesh.cells[0].data + k)
        k += mesh.points.shape[0]
    cells = np.concatenate(cells)

    meshio.write_points_cells(
        filename, points, {"triangle": cells}, point_data={"srgb1": srgb1_vals}
    )
