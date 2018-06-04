# -*- coding: utf-8 -*-
#
import numpy
import sympy

from ..line_segment import tree_alp


def tree_sph(polar, azimuthal, n, standardization, symbolic=False):
    """Evaluate all spherical harmonics of degree at most `n` at angles `polar`,
    `azimuthal`.
    """
    cos = numpy.vectorize(sympy.cos) if symbolic else numpy.cos

    # Conventions from
    # <https://en.wikipedia.org/wiki/Spherical_harmonics#Orthogonality_and_normalization>.
    config = {
        "acoustic": ("complex spherical", False),
        "quantum mechanic": ("complex spherical", True),
        "geodetic": ("complex spherical 1", False),
        "schmidt": ("schmidt", False),
    }

    standard, cs_phase = config[standardization]

    return tree_alp(
        cos(polar),
        n,
        phi=azimuthal,
        standardization=standard,
        with_condon_shortley_phase=cs_phase,
        symbolic=symbolic,
    )
