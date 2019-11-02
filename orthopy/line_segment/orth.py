import itertools

import numpy
import sympy

from ..tools import line_tree
from . import recurrence_coefficients


def tree_chebyshev1(X, n, standardization, symbolic=False):
    one_half = sympy.S(1) / 2 if symbolic else 0.5
    return tree_jacobi(X, n, -one_half, -one_half, standardization, symbolic=symbolic)


def tree_chebyshev2(X, n, standardization, symbolic=False):
    one_half = sympy.S(1) / 2 if symbolic else 0.5
    return tree_jacobi(X, n, +one_half, +one_half, standardization, symbolic=symbolic)


def tree_legendre(X, n, standardization, symbolic=False):
    return tree_jacobi(X, n, 0, 0, standardization, symbolic=symbolic)


def tree_gegenbauer(X, n, lmbda, standardization, symbolic=False):
    return tree_jacobi(X, n, lmbda, lmbda, standardization, symbolic=symbolic)


def tree_jacobi(X, n, alpha, beta, standardization, symbolic=False):
    args = recurrence_coefficients.jacobi(
        n, alpha, beta, standardization, symbolic=symbolic
    )
    return line_tree(X, *args)


class _Natural:
    def __init__(self, x, symbolic):
        self.frac = sympy.Rational if symbolic else lambda x, y: x / y
        sqrt = numpy.vectorize(sympy.sqrt) if symbolic else numpy.sqrt

        self.p0 = 1
        self.sqrt1mx2 = sqrt(1 - x ** 2)
        return

    def z0_factor(self, L):
        return self.sqrt1mx2 / (2 * L)

    def z1_factor(self, L):
        return self.sqrt1mx2 * (2 * L - 1)

    def C0(self, L):
        return [self.frac(2 * L - 1, L - m) for m in range(-L + 1, L)]

    def C1(self, L):
        return [self.frac(L - 1 + m, L - m) for m in range(-L + 2, L - 1)]


class _Spherical:
    def __init__(self, x, symbolic):
        self.frac = sympy.Rational if symbolic else lambda x, y: x / y
        self.sqrt = numpy.vectorize(sympy.sqrt) if symbolic else numpy.sqrt
        pi = sympy.pi if symbolic else numpy.pi

        self.p0 = 1 / self.sqrt(4 * pi)
        self.sqrt1mx2 = self.sqrt(1 - x ** 2)
        return

    def z0_factor(self, L):
        return self.sqrt1mx2 * self.sqrt(self.frac(2 * L + 1, 2 * L))

    def z1_factor(self, L):
        return self.sqrt1mx2 * self.sqrt(self.frac(2 * L + 1, 2 * L))

    def C0(self, L):
        m = numpy.arange(-L + 1, L)
        d = (L + m) * (L - m)
        return self.sqrt((2 * L - 1) * (2 * L + 1)) / self.sqrt(d)

    def C1(self, L):
        m = numpy.arange(-L + 2, L - 1)
        d = (L + m) * (L - m)
        return self.sqrt((L + m - 1) * (L - m - 1) * (2 * L + 1)) / self.sqrt(
            (2 * L - 3) * d
        )


class _ComplexSpherical:
    def __init__(self, x, phi, symbolic, geodesic):
        pi = sympy.pi if symbolic else numpy.pi
        imag_unit = sympy.I if symbolic else 1j
        self.sqrt = numpy.vectorize(sympy.sqrt) if symbolic else numpy.sqrt
        self.frac = sympy.Rational if symbolic else lambda x, y: x / y
        exp = sympy.exp if symbolic else numpy.exp

        # The starting value 1 has the effect of multiplying the entire tree by
        # sqrt(4*pi). This convention is used in geodesy and and spectral
        # analysis.
        self.p0 = 1 if geodesic else 1 / self.sqrt(4 * pi)
        self.exp_iphi = exp(imag_unit * phi)
        self.sqrt1mx2 = self.sqrt(1 - x ** 2)
        return

    def z0_factor(self, L):
        return self.sqrt1mx2 * self.sqrt(self.frac(2 * L + 1, 2 * L)) / self.exp_iphi

    def z1_factor(self, L):
        return self.sqrt1mx2 * self.sqrt(self.frac(2 * L + 1, 2 * L)) * self.exp_iphi

    def C0(self, L):
        m = numpy.arange(-L + 1, L)
        d = (L + m) * (L - m)
        return self.sqrt((2 * L - 1) * (2 * L + 1)) / self.sqrt(d)

    def C1(self, L):
        m = numpy.arange(-L + 2, L - 1)
        d = (L + m) * (L - m)
        return self.sqrt((L + m - 1) * (L - m - 1) * (2 * L + 1)) / self.sqrt(
            (2 * L - 3) * d
        )


class _Normal:
    def __init__(self, x, symbolic):
        self.sqrt = numpy.vectorize(sympy.sqrt) if symbolic else numpy.sqrt
        self.frac = sympy.Rational if symbolic else lambda x, y: x / y

        self.p0 = 1 / self.sqrt(2)
        self.sqrt1mx2 = self.sqrt(1 - x ** 2)
        return

    def z0_factor(self, L):
        return self.sqrt1mx2 * self.sqrt(self.frac(2 * L + 1, 2 * L))

    def z1_factor(self, L):
        return self.sqrt1mx2 * self.sqrt(self.frac(2 * L + 1, 2 * L))

    def C0(self, L):
        m = numpy.arange(-L + 1, L)
        d = (L + m) * (L - m)
        return self.sqrt((2 * L - 1) * (2 * L + 1)) / self.sqrt(d)

    def C1(self, L):
        m = numpy.arange(-L + 2, L - 1)
        d = (L + m) * (L - m)
        return self.sqrt((L + m - 1) * (L - m - 1) * (2 * L + 1)) / self.sqrt(
            (2 * L - 3) * d
        )


class _Schmidt:
    def __init__(self, x, phi, symbolic):
        self.sqrt = numpy.vectorize(sympy.sqrt) if symbolic else numpy.sqrt
        self.frac = sympy.Rational if symbolic else lambda x, y: x / y

        if phi is None:
            self.p0 = 2
            self.exp_iphi = 1
        else:
            self.p0 = 1
            imag_unit = sympy.I if symbolic else 1j
            exp = sympy.exp if symbolic else numpy.exp
            self.exp_iphi = exp(imag_unit * phi)

        self.sqrt1mx2 = self.sqrt(1 - x ** 2)
        return

    def z0_factor(self, L):
        return self.sqrt1mx2 * self.sqrt(self.frac(2 * L - 1, 2 * L)) / self.exp_iphi

    def z1_factor(self, L):
        return self.sqrt1mx2 * self.sqrt(self.frac(2 * L - 1, 2 * L)) * self.exp_iphi

    def C0(self, L):
        m = numpy.arange(-L + 1, L)
        d = self.sqrt((L + m) * (L - m))
        return (2 * L - 1) / d

    def C1(self, L):
        m = numpy.arange(-L + 2, L - 1)
        d = self.sqrt((L + m) * (L - m))
        return self.sqrt((L + m - 1) * (L - m - 1)) / d


def tree_alp(
    x, n, standardization, phi=None, with_condon_shortley_phase=True, symbolic=False
):
    """Evaluates the entire tree of associated Legendre polynomials up to depth n.

    There are many recurrence relations that can be used to construct the associated
    Legendre polynomials. However, only few are numerically stable.  Many
    implementations (including this one) use the classical Legendre recurrence relation
    with increasing L.

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

    The return value is a list of arrays, where `out[k]` hosts the `2*k+1` values of the
    `k`th level of the tree

                              (0, 0)
                    (-1, 1)   (0, 1)   (1, 1)
          (-2, 2)   (-1, 2)   (0, 2)   (1, 2)   (2, 2)
            ...       ...       ...     ...       ...
    """
    return list(
        itertools.islice(
            IteratorAlp(x, standardization, phi, with_condon_shortley_phase, symbolic),
            n + 1,
        )
    )


class IteratorAlp:
    def __init__(
        self,
        x,
        standardization,
        phi=None,
        with_condon_shortley_phase=True,
        symbolic=False,
    ):
        # assert numpy.all(numpy.abs(x) <= 1.0)
        d = {
            "natural": (_Natural, [x, symbolic]),
            "spherical": (_Spherical, [x, symbolic]),
            "complex spherical": (_ComplexSpherical, [x, phi, symbolic, False]),
            "complex spherical 1": (_ComplexSpherical, [x, phi, symbolic, True]),
            "normal": (_Normal, [x, symbolic]),
            "schmidt": (_Schmidt, [x, phi, symbolic]),
        }
        fun, args = d[standardization]
        self.c = fun(*args)

        if with_condon_shortley_phase:

            def z1_factor_CSP(L):
                return -1 * self.c.z1_factor(L)

        else:
            z1_factor_CSP = self.c.z1_factor

        self.z1_factor_CSP = z1_factor_CSP

        self.k = 0
        self.x = x
        self.last = [None, None]

    def __iter__(self):
        return self

    def __next__(self):

        # Here comes the actual loop.
        e = numpy.ones_like(self.x, dtype=int)
        if self.k == 0:
            out = numpy.array([e * self.c.p0])
        else:
            [self.last[0][0] * self.c.z0_factor(self.k)]
            out = numpy.concatenate(
                [
                    [self.last[0][0] * self.c.z0_factor(self.k)],
                    self.last[0] * numpy.multiply.outer(self.c.C0(self.k), self.x),
                    [self.last[0][-1] * self.z1_factor_CSP(self.k)],
                ]
            )

            if self.k > 1:
                out[2:-2] -= numpy.multiply.outer(self.c.C1(self.k), e) * self.last[1]

        self.last[1] = self.last[0]
        self.last[0] = out
        self.k += 1
        return out
