# -*- coding: utf-8 -*-
#
import numpy

from .alp import alp_tree


def sph_tree(n, theta, phi):
    return alp_tree(
        n, numpy.cos(theta), phi,
        normalization='complex spherical'
        )
