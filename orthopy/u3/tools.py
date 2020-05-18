import numpy


def write(filename, f, n=5, colors_enhancement=2.5):
    """Write a function `f` defined in terms of spherical coordinates to a file.
    """
    import meshio
    import meshzoo
    import cplot

    points, cells = meshzoo.icosa_sphere(n)
    # get spherical coordinates from points
    polar = numpy.arccos(points[:, 2])
    azimuthal = numpy.arctan2(points[:, 1], points[:, 0])
    vals = cplot.get_srgb1(f(polar, azimuthal), colorspace="cam16")

    # exaggerate colors a bit
    vals *= colors_enhancement
    vals[vals > 1] = 1
    vals[vals < 0] = 0

    meshio.write_points_cells(
        filename, points, {"triangle": cells}, point_data={"srgb1": vals}
    )
