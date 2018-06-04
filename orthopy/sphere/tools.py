# -*- coding: utf-8 -*-
#
import numpy


def write(filename, f):
    """Write a function `f` defined in terms of spherical coordinates to a file.
    """
    import meshio
    import meshzoo

    points, cells = meshzoo.iso_sphere(5)
    # get spherical coordinates from points
    polar = numpy.arccos(points[:, 2])
    azimuthal = numpy.arctan2(points[:, 1], points[:, 0])
    vals = f(polar, azimuthal)
    meshio.write(filename, points, {"triangle": cells}, point_data={"f": vals})
    return
