# -*- coding: utf-8 -*-
#
from __future__ import division

import numpy
import sympy


# pylint: disable=too-many-locals
def tree(n, bary, standardization, symbolic=False):
    '''Evaluates the entire tree of orthogonal triangle polynomials.

    The return value is a list of arrays, where `out[k]` hosts the `2*k+1`
    values of the `k`th level of the tree

        (0, 0)
        (0, 1)   (1, 1)
        (0, 2)   (1, 2)   (2, 2)
          ...      ...      ...

    For reference, see

    Abedallah Rababah,
    Recurrence Relations for Orthogonal Polynomials on Triangular Domains,
    Mathematics 2016, 4(2), 25,
    <https://doi.org/10.3390/math4020025>.
    '''
    frac = sympy.Rational if symbolic else lambda x, y: x/y
    sqrt = sympy.sqrt if symbolic else numpy.sqrt

    if standardization == '1':
        p0 = 1

        def alpha(n):
            return numpy.array([
                frac(n*(2*n+1), (n-r) * (n+r+1))
                for r in range(n)
                ])

        def beta(n):
            return numpy.array([
                frac(n * (2*r+1)**2, (n-r) * (n+r+1) * (2*n-1))
                for r in range(n)
                ])

        def gamma(n):
            return numpy.array([frac(
                (n-r-1) * (n+r) * (2*n+1), (n-r) * (n+r+1) * (2*n-1)
                ) for r in range(n-1)
                ])

        def delta(n):
            return frac(2*n-1, n)

        def epsilon(n):
            return frac(n-1, n)

    else:
        # The coefficients here are based on the insight that
        #
        #   int_T P_{n, r}^2 =
        #       int_0^1 L_r^2(t) dt * int_0^1 q_{n,r}^2(w) (1-w)^(r+s+1) dw.
        #
        # For reference, see
        # page 219 (and the reference to Gould, 1972) in
        #
        #  Farouki, Goodman, Sauer,
        #  Construction of orthogonal bases for polynomials in Bernstein form
        #  on triangular and simplex domains,
        #  Computer Aided Geometric Design 20 (2003) 209–230.
        #
        # From this, one gets
        #
        #   int_T P_{n, r}^2 = 1 / (2*r+1) / (2*n+2)
        #       sum_{i=0}^{n-r} sum_{j=0}^{n-r}
        #           (-1)**(i+j) * binom(n+r+1, i) * binom(n-r, i)
        #                       * binom(n+r+1, j) * binom(n-r, j)
        #                       / binom(2*n+1, i+j)
        #
        # The Legendre integral is 1/(2*r+1) and, astonishingly, the double sum
        # is always 1, hence
        #
        #   int_T P_{n, r}^2 = 1 / (2*r+1) / (2*n+2).
        #
        assert standardization == 'normal'
        p0 = sqrt(2)

        def alpha(n):
            return numpy.array([
                frac(2*n+1, (n-r) * (n+r+1)) * sqrt((n+1)*n)
                for r in range(n)
                ])

        def beta(n):
            return numpy.array([
                frac((2*r+1)**2, (n-r) * (n+r+1) * (2*n-1))
                * sqrt((n+1)*n)
                for r in range(n)
                ])

        def gamma(n):
            return numpy.array([frac(
                (n-r-1) * (n+r) * (2*n+1),
                (n-r) * (n+r+1) * (2*n-1)
                ) for r in range(n-1)
                ]) * sqrt(frac(n+1, n-1))

        def delta(n):
            return sqrt(frac((2*n+1) * (n+1) * (2*n-1), n**3))

        def epsilon(n):
            return sqrt(frac(
                (2*n+1) * (n+1) * (n-1), (2*n-3) * n**2
                ))

    out = [numpy.array([numpy.full(bary.shape[1:], p0)])]

    u, v, w = bary

    for L in range(1, n+1):
        out.append(numpy.concatenate([
            out[L-1] * (numpy.multiply.outer(alpha(L), 1-2*w).T - beta(L)).T,
            [out[L-1][L-1] * (u-v) * delta(L)],
            ])
            )

        if L > 1:
            out[-1][:L-1] -= (out[L-2].T * gamma(L)).T
            out[-1][-1] -= out[L-2][L-2] * (u+v)**2 * epsilon(L)

    return out
