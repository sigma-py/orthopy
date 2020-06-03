import itertools

import numpy
import sympy

from ..helpers import Eval135


def tree(n, *args, **kwargs):
    return list(itertools.islice(Eval(*args, **kwargs), n + 1))


class Eval(Eval135):
    """Evaluate spherical harmonics degree by degree `n` at angles `polar`, `azimuthal`.
    """

    def __init__(self, polar, azimuthal, scaling, symbolic=False):
        # Conventions from
        # <https://en.wikipedia.org/wiki/Spherical_harmonics#Orthogonality_and_normalization>.
        rc = {
            "acoustic": RCComplexSpherical(azimuthal, False, symbolic, geodetic=False),
            "quantum mechanic": RCComplexSpherical(
                azimuthal, True, symbolic, geodetic=False
            ),
            "geodetic": RCComplexSpherical(azimuthal, False, symbolic, geodetic=True),
            "schmidt": RCSchmidt(azimuthal, False, symbolic),
        }[scaling]

        cos = numpy.vectorize(sympy.cos) if symbolic else numpy.cos
        super().__init__(rc, cos(polar), symbolic=symbolic)


class RCSpherical:
    def __init__(self, with_cs_phase, symbolic):
        self.S = sympy.S if symbolic else lambda x: x
        self.sqrt = numpy.vectorize(sympy.sqrt) if symbolic else numpy.sqrt
        pi = sympy.pi if symbolic else numpy.pi

        self.p0 = 1 / self.sqrt(4 * pi)
        self.phase = -1 if with_cs_phase else 1

    def __getitem__(self, L):
        L = self.S(L)

        z0 = self.sqrt((2 * L + 1) / (2 * L))
        z1 = z0 * self.phase
        #
        m = numpy.arange(-L + 1, L)
        c0 = self.sqrt((2 * L - 1) * (2 * L + 1) / ((L + m) * (L - m)))
        #
        if L == 1:
            c1 = None
        else:
            m = numpy.arange(-L + 2, L - 1)
            c1 = self.sqrt(
                (L + m - 1)
                * (L - m - 1)
                * (2 * L + 1)
                / ((2 * L - 3) * (L + m) * (L - m))
            )
        return z0, z1, c0, c1


class RCComplexSpherical:
    def __init__(self, phi, with_cs_phase, symbolic, geodetic):
        pi = sympy.pi if symbolic else numpy.pi
        imag_unit = sympy.I if symbolic else 1j
        self.sqrt = numpy.vectorize(sympy.sqrt) if symbolic else numpy.sqrt
        self.frac = numpy.vectorize(sympy.Rational) if symbolic else lambda x, y: x / y
        exp = sympy.exp if symbolic else numpy.exp

        # The starting value 1 has the effect of multiplying the entire tree by
        # sqrt(4*pi). This convention is used in geodesy and and spectral
        # analysis.
        self.p0 = 1 if geodetic else 1 / self.sqrt(4 * pi)
        self.exp_iphi = exp(imag_unit * phi)
        self.phase = -1 if with_cs_phase else 1

    def __getitem__(self, L):
        z0 = self.sqrt(self.frac(2 * L + 1, 2 * L)) / self.exp_iphi
        z1 = self.sqrt(self.frac(2 * L + 1, 2 * L)) * self.exp_iphi * self.phase
        #
        m = numpy.arange(-L + 1, L)
        c0 = self.sqrt(self.frac((2 * L - 1) * (2 * L + 1), (L + m) * (L - m)))
        #
        if L == 1:
            c1 = None
        else:
            m = numpy.arange(-L + 2, L - 1)
            c1 = self.sqrt(
                self.frac(
                    (L + m - 1) * (L - m - 1) * (2 * L + 1),
                    (2 * L - 3) * (L + m) * (L - m),
                )
            )
        return z0, z1, c0, c1


class RCSchmidt:
    def __init__(self, phi, with_cs_phase, symbolic):
        self.sqrt = numpy.vectorize(sympy.sqrt) if symbolic else numpy.sqrt
        self.frac = numpy.vectorize(sympy.Rational) if symbolic else lambda x, y: x / y
        self.phase = -1 if with_cs_phase else 1

        if phi is None:
            self.p0 = 2
            self.exp_iphi = 1
        else:
            self.p0 = 1
            imag_unit = sympy.I if symbolic else 1j
            exp = sympy.exp if symbolic else numpy.exp
            self.exp_iphi = exp(imag_unit * phi)

    def __getitem__(self, L):
        z0 = self.sqrt(self.frac(2 * L - 1, 2 * L)) / self.exp_iphi
        z1 = self.sqrt(self.frac(2 * L - 1, 2 * L)) * self.exp_iphi * self.phase
        #
        m = numpy.arange(-L + 1, L)
        c0 = (2 * L - 1) / self.sqrt((L + m) * (L - m))
        #
        if L == 1:
            c1 = None
        else:
            m = numpy.arange(-L + 2, L - 1)
            c1 = self.sqrt(self.frac((L + m - 1) * (L - m - 1), (L + m) * (L - m)))
        return z0, z1, c0, c1
