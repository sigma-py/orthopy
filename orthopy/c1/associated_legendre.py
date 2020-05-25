import itertools

import numpy
import sympy

from ..helpers import Iterator135


def tree(n, *args, **kwargs):
    return list(itertools.islice(Iterator(*args, **kwargs), n + 1))


class Iterator(Iterator135):
    """
    Useful references are

    Taweetham Limpanuparb, Josh Milthorpe,
    Associated Legendre Polynomials and Spherical Harmonics Computation for Chemistry
    Applications,
    Proceedings of The 40th Congress on Science and Technology of Thailand;
    2014 Dec 2-4, Khon Kaen, Thailand. P. 233-241.
    <https://arxiv.org/abs/1410.1748>

    and

    Schneider et al.,
    A new Fortran 90 program to compute regular and irregular associated Legendre
    functions,
    Computer Physics Communications,
    Volume 181, Issue 12, December 2010, Pages 2091-2097,
    <https://doi.org/10.1016/j.cpc.2010.08.038>.
    """

    def __init__(self, X, scaling, symbolic=False):
        cls = {"classical": RCClassical, "normal": RCNormal}[scaling]
        rc = cls(symbolic)
        super().__init__(rc, X, symbolic)


class RCClassical:
    def __init__(self, symbolic):
        self.frac = sympy.Rational if symbolic else lambda x, y: x / y
        self.p0 = 1

    def __getitem__(self, L):
        z0 = self.frac(1, 2 * L)
        z1 = -(2 * L - 1)
        c0 = [self.frac(2 * L - 1, L - m) for m in range(-L + 1, L)]
        if L == 1:
            c1 = None
        else:
            c1 = [self.frac(L - 1 + m, L - m) for m in range(-L + 2, L - 1)]
        return z0, z1, c0, c1


class RCNormal:
    def __init__(self, symbolic):
        self.sqrt = numpy.vectorize(sympy.sqrt) if symbolic else numpy.sqrt
        self.frac = numpy.vectorize(sympy.Rational) if symbolic else lambda x, y: x / y

        self.p0 = 1 / self.sqrt(2)

    def __getitem__(self, L):
        z0 = self.sqrt(self.frac(2 * L + 1, 2 * L))
        z1 = -self.sqrt(self.frac(2 * L + 1, 2 * L))
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
