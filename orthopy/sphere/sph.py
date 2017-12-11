# -*- coding: utf-8 -*-
#
import numpy
import sympy

from .. import line


def sph_tree(n, theta, phi, symbolic=False):
    cos = numpy.vectorize(sympy.cos) if symbolic else numpy.cos
    return line.alp_tree(
        n, cos(theta), phi,
        normalization='complex spherical',
        symbolic=symbolic
        )
