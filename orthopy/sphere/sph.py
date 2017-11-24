# -*- coding: utf-8 -*-
#
import numpy

from .. import line


def sph_tree(n, theta, phi):
    return line.alp_tree(
        n, numpy.cos(theta), phi,
        normalization='complex spherical'
        )
