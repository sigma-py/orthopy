import itertools

import numpy
import sympy

from ..line_segment import IteratorAlp


def tree_sph(polar, azimuthal, n, standardization, symbolic=False):
    """Evaluate all spherical harmonics of degree at most `n` at angles `polar`,
    `azimuthal`.
    """
    return list(
        itertools.islice(Iterator(polar, azimuthal, standardization, symbolic), n + 1)
    )


class Iterator(IteratorAlp):
    def __init__(self, polar, azimuthal, standardization, symbolic=False):
        cos = numpy.vectorize(sympy.cos) if symbolic else numpy.cos

        # Conventions from
        # <https://en.wikipedia.org/wiki/Spherical_harmonics#Orthogonality_and_normalization>.
        standard, cs_phase = {
            "acoustic": ("complex spherical", False),
            "quantum mechanic": ("complex spherical", True),
            "geodetic": ("complex spherical 1", False),
            "schmidt": ("schmidt", False),
        }[standardization]

        super().__init__(
            cos(polar),
            phi=azimuthal,
            standardization=standard,
            with_condon_shortley_phase=cs_phase,
            symbolic=symbolic,
        )
