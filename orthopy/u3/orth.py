import itertools

import numpy
import sympy

from ..c1 import associated_legendre


def tree_sph(n, *args, **kwargs):
    return list(itertools.islice(Iterator(*args, **kwargs), n + 1))


class Iterator(associated_legendre.Iterator):
    """Evaluate all spherical harmonics of degree at most `n` at angles `polar`,
    `azimuthal`.
    """

    def __init__(self, polar, azimuthal, scaling, symbolic=False):
        cos = numpy.vectorize(sympy.cos) if symbolic else numpy.cos

        # Conventions from
        # <https://en.wikipedia.org/wiki/Spherical_harmonics#Orthogonality_and_normalization>.
        standard, cs_phase = {
            "acoustic": ("complex spherical", False),
            "quantum mechanic": ("complex spherical", True),
            "geodetic": ("complex spherical 1", False),
            "schmidt": ("schmidt", False),
        }[scaling]

        super().__init__(
            cos(polar),
            phi=azimuthal,
            scaling=standard,
            with_condon_shortley_phase=cs_phase,
            symbolic=symbolic,
        )
