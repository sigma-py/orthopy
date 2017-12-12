# -*- coding: utf-8 -*-
#
import numpy
import sympy

from .. import line


def sph_tree(
        n, polar_angle, azimuthal_angle, normalization=None, symbolic=False
        ):
    '''Evaluate all spherical harmonics of degree at most `n` at `polar_angle`,
    `azimuthal_angle`.
    '''
    cos = numpy.vectorize(sympy.cos) if symbolic else numpy.cos
    # 'acoustic'
    # 'quantum mechanic'
    # 'geodesic'
    # 'schmidt/racah'
    return line.alp_tree(
        n, cos(polar_angle), azimuthal_angle,
        normalization='complex spherical',
        symbolic=symbolic
        )
