import itertools

from .main import Eval


def write_single(filename, n, r, scaling, res=20, colors_enhancement=2.5):
    import meshio
    import meshzoo
    import cplot

    points, cells = meshzoo.icosa_sphere(res)

    evaluator = Eval(points.T, scaling, complex_valued=True)
    vals = next(itertools.islice(evaluator, n, None))[r]

    srgb1_vals = cplot.get_srgb1(vals, colorspace="cam16")

    # exaggerate colors a bit
    srgb1_vals *= colors_enhancement
    srgb1_vals[srgb1_vals > 1] = 1
    srgb1_vals[srgb1_vals < 0] = 0

    meshio.write_points_cells(
        filename, points, {"triangle": cells}, point_data={"srgb1": srgb1_vals}
    )


# def write_tree(filename, f, res=5, colors_enhancement=2.5):
#     import meshio
#     import meshzoo
#     import cplot
#
#     points, cells = meshzoo.icosa_sphere(res)
#     # get spherical coordinates from points
#     polar = numpy.arccos(points[:, 2])
#     azimuthal = numpy.arctan2(points[:, 1], points[:, 0])
#     vals = cplot.get_srgb1(f(polar, azimuthal), colorspace="cam16")
#
#     # exaggerate colors a bit
#     vals *= colors_enhancement
#     vals[vals > 1] = 1
#     vals[vals < 0] = 0
#
#     meshio.write_points_cells(
#         filename, points, {"triangle": cells}, point_data={"srgb1": vals}
#     )
