# -*- coding: utf-8 -*-
#
from __future__ import division

import numpy
import sympy

from . import recurrence_coefficients

from ..tools import line_tree


def tree_chebyshev1(X, n, standardization, symbolic=False):
    one_half = sympy.S(1)/2 if symbolic else 0.5
    return tree_jacobi(
        X, n, -one_half, -one_half, standardization, symbolic=symbolic
        )


def tree_chebyshev2(X, n, standardization, symbolic=False):
    one_half = sympy.S(1)/2 if symbolic else 0.5
    return tree_jacobi(
        X, n, +one_half, +one_half, standardization, symbolic=symbolic
        )


def tree_gegenbauer(X, n, lmbda, standardization, symbolic=False):
    return tree_jacobi(X, n, lmbda, lmbda, standardization, symbolic=symbolic)


def tree_legendre(X, n, standardization, symbolic=False):
    return tree_jacobi(X, n, 0, 0, standardization, symbolic=symbolic)


# pylint: disable=too-many-arguments
def tree_jacobi(X, n, alpha, beta, standardization, symbolic=False):
    args = recurrence_coefficients.jacobi(
            n, alpha, beta, standardization, symbolic=symbolic
            )
    return line_tree(X, *args)


def tree_alp(
        x, n, standardization, phi=None,
        with_condon_shortley_phase=True,
        symbolic=False
        ):
    '''Evaluates the entire tree of associated Legendre polynomials up to depth
    n.
    There are many recurrence relations that can be used to construct the
    associated Legendre polynomials. However, only few are numerically stable.
    Many implementations (including this one) use the classical Legendre
    recurrence relation with increasing L.

    Useful references are

    Taweetham Limpanuparb, Josh Milthorpe,
    Associated Legendre Polynomials and Spherical Harmonics Computation for
    Chemistry Applications,
    Proceedings of The 40th Congress on Science and Technology of Thailand;
    2014 Dec 2-4, Khon Kaen, Thailand. P. 233-241.
    <https://arxiv.org/abs/1410.1748>

    and

    Schneider et al.,
    A new Fortran 90 program to compute regular and irregular associated
    Legendre functions,
    Computer Physics Communications,
    Volume 181, Issue 12, December 2010, Pages 2091-2097,
    <https://doi.org/10.1016/j.cpc.2010.08.038>.

    The return value is a list of arrays, where `out[k]` hosts the `2*k+1`
    values of the `k`th level of the tree

                              (0, 0)
                    (-1, 1)   (0, 1)   (1, 1)
          (-2, 2)   (-1, 2)   (0, 2)   (1, 2)   (2, 2)
            ...       ...       ...     ...       ...
    '''
    # pylint: disable=too-many-statements,too-many-locals
    # assert numpy.all(numpy.abs(x) <= 1.0)

    exp = sympy.exp if symbolic else numpy.exp
    pi = sympy.pi if symbolic else numpy.pi
    sqrt = numpy.vectorize(sympy.sqrt) if symbolic else numpy.sqrt
    frac = sympy.Rational if symbolic else lambda x, y: x/y
    imag_unit = sympy.I if symbolic else 1j

    sqrt1mx2 = sqrt(1 - x**2)

    e = numpy.ones_like(x, dtype=int)

    if standardization == 'natural':
        alpha = 1

        def z0_factor(L):
            return sqrt1mx2 / (2*L)

        def z1_factor(L):
            return sqrt1mx2 * (2*L-1)

        def C0(L):
            return [frac(2*L-1, L-m) for m in range(-L+1, L)]

        def C1(L):
            return [frac(L-1+m, L-m) for m in range(-L+2, L-1)]

    elif standardization == 'spherical':
        alpha = 1 / sqrt(4*pi)

        def z0_factor(L):
            return sqrt1mx2 * sqrt(frac(2*L+1, 2*L))

        def z1_factor(L):
            return sqrt1mx2 * sqrt(frac(2*L+1, 2*L))

        def C0(L):
            m = numpy.arange(-L+1, L)
            d = (L+m) * (L-m)
            return sqrt((2*L-1) * (2*L+1)) / sqrt(d)

        def C1(L):
            m = numpy.arange(-L+2, L-1)
            d = (L+m) * (L-m)
            return sqrt((L+m-1) * (L-m-1) * (2*L+1)) / sqrt((2*L-3) * d)

    elif standardization in ['complex spherical', 'complex spherical 1']:
        # The starting value 1 has the effect of multiplying the entire tree by
        # sqrt(4*pi). This convention is used in geodesy and and spectral
        # analysis.
        alpha = 1 / sqrt(4*pi) if standardization == 'complex spherical' else 1

        exp_iphi = exp(imag_unit * phi)

        def z0_factor(L):
            return sqrt1mx2 * sqrt(frac(2*L+1, 2*L)) / exp_iphi

        def z1_factor(L):
            return sqrt1mx2 * sqrt(frac(2*L+1, 2*L)) * exp_iphi

        def C0(L):
            m = numpy.arange(-L+1, L)
            d = (L+m) * (L-m)
            return sqrt((2*L-1) * (2*L+1)) / sqrt(d)

        def C1(L):
            m = numpy.arange(-L+2, L-1)
            d = (L+m) * (L-m)
            return sqrt((L+m-1) * (L-m-1) * (2*L+1)) / sqrt((2*L-3) * d)

    elif standardization == 'normal':
        alpha = 1 / sqrt(2)

        def z0_factor(L):
            return sqrt1mx2 * sqrt(frac(2*L+1, 2*L))

        def z1_factor(L):
            return sqrt1mx2 * sqrt(frac(2*L+1, 2*L))

        def C0(L):
            m = numpy.arange(-L+1, L)
            d = (L+m) * (L-m)
            return sqrt((2*L-1) * (2*L+1)) / sqrt(d)

        def C1(L):
            m = numpy.arange(-L+2, L-1)
            d = (L+m) * (L-m)
            return sqrt((L+m-1) * (L-m-1) * (2*L+1)) / sqrt((2*L-3) * d)

    else:
        assert standardization == 'schmidt', \
            'Unknown standardization \'{}\'.'.format(standardization)

        if phi is None:
            alpha = 2
            exp_iphi = 1
        else:
            alpha = 1
            exp_iphi = exp(imag_unit * phi)

        def z0_factor(L):
            return sqrt1mx2 * sqrt(frac(2*L-1, 2*L)) / exp_iphi

        def z1_factor(L):
            return sqrt1mx2 * sqrt(frac(2*L-1, 2*L)) * exp_iphi

        def C0(L):
            m = numpy.arange(-L+1, L)
            d = sqrt((L+m) * (L-m))
            return (2*L-1) / d

        def C1(L):
            m = numpy.arange(-L+2, L-1)
            d = sqrt((L+m) * (L-m))
            return sqrt((L+m-1) * (L-m-1)) / d

    if with_condon_shortley_phase:
        def z1_factor_CSP(L):
            return -1 * z1_factor(L)
    else:
        z1_factor_CSP = z1_factor

    # Here comes the actual loop.
    out = [[e * alpha]]
    for L in range(1, n+1):
        out.append(
            numpy.concatenate([
                [out[L-1][0] * z0_factor(L)],
                out[L-1] * numpy.multiply.outer(C0(L), x),
                [out[L-1][-1] * z1_factor_CSP(L)],
                ])
            )

        if L > 1:
            out[-1][2:-2] -= numpy.multiply.outer(C1(L), e) * out[L-2]

    return out
