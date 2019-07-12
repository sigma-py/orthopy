import numpy


def write(filename, f):
    """Write a function `f` defined in terms of spherical coordinates to a file.
    """
    import meshio
    import meshzoo
    import cplot

    points, cells = meshzoo.iso_sphere(5)
    # get spherical coordinates from points
    polar = numpy.arccos(points[:, 2])
    azimuthal = numpy.arctan2(points[:, 1], points[:, 0])
    vals = cplot.get_srgb1(f(polar, azimuthal), colorspace="cam16")

    vals *= 2.5
    vals[vals > 1] = 1
    vals[vals < 0] = 0

    meshio.write_points_cells(
        filename, points, {"triangle": cells}, point_data={"srgb1": vals}
    )
    return
