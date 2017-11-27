# -*- coding: utf-8 -*-
#
import numpy
import sympy

from .. import line


def sph_tree(n, theta, phi):
    cos = numpy.vectorize(sympy.cos)
    return line.alp_tree(
        n, cos(theta), phi,
        normalization='complex spherical'
        )
