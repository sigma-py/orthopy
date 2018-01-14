# -*- coding: utf-8 -*-
#
from __future__ import division

import numpy
import sympy


# pylint: disable=too-many-locals
def tree(bary, n, standardization, symbolic=False):
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
    S = numpy.vectorize(sympy.S) if symbolic else lambda x: x
    sqrt = numpy.vectorize(sympy.sqrt) if symbolic else numpy.sqrt

    if standardization == '1':
        p0 = 1

        def alpha(n):
            r = numpy.arange(n)
            return S(n*(2*n+1)) / ((n-r) * (n+r+1))

        def beta(n):
            r = numpy.arange(n)
            return S(n * (2*r+1)**2) / ((n-r) * (n+r+1) * (2*n-1))

        def gamma(n):
            r = numpy.arange(n-1)
            return S((n-r-1) * (n+r) * (2*n+1)) / ((n-r) * (n+r+1) * (2*n-1))

        def delta(n):
            return S(2*n-1) / n

        def epsilon(n):
            return S(n-1) / n

    else:
        # The coefficients here are based on the insight that
        #
        #   int_T P_{n, r}^2 =
        #       int_0^1 L_r^2(t) dt * int_0^1 q_{n,r}(w)^2 (1-w)^(r+s+1) dw.
        #
        # For reference, see
        # page 219 (and the reference to Gould, 1972) in
        #
        #  Farouki, Goodman, Sauer,
        #  Construction of orthogonal bases for polynomials in Bernstein form
        #  on triangular and simplex domains,
        #  Computer Aided Geometric Design 20 (2003) 209–230.
        #
        # The Legendre integral is 1/(2*r+1), and one gets
        #
        #   int_T P_{n, r}^2 = 1 / (2*r+1) / (2*n+2)
        #       sum_{i=0}^{n-r} sum_{j=0}^{n-r}
        #           (-1)**(i+j) * binom(n+r+1, i) * binom(n-r, i)
        #                       * binom(n+r+1, j) * binom(n-r, j)
        #                       / binom(2*n+1, i+j)
        #
        # Astonishingly, the double sum is always 1, hence
        #
        #   int_T P_{n, r}^2 = 1 / (2*r+1) / (2*n+2).
        #
        assert standardization == 'normal'
        p0 = sqrt(2)

        def alpha(n):
            r = numpy.arange(n)
            return sqrt((n+1)*n) * (S(2*n+1) / ((n-r) * (n+r+1)))

        def beta(n):
            r = numpy.arange(n)
            return sqrt((n+1)*n) * S((2*r+1)**2) / ((n-r) * (n+r+1) * (2*n-1))

        def gamma(n):
            r = numpy.arange(n-1)
            return sqrt(S(n+1) / (n-1)) * (
                S((n-r-1) * (n+r) * (2*n+1)) /
                ((n-r) * (n+r+1) * (2*n-1))
                )

        def delta(n):
            return sqrt(S((2*n+1) * (n+1) * (2*n-1)) / n**3)

        def epsilon(n):
            return sqrt(S((2*n+1) * (n+1) * (n-1)) / ((2*n-3) * n**2))

    u, v, w = bary

    out = [numpy.array([numpy.zeros_like(u) + p0])]

    for L in range(1, n+1):
        out.append(numpy.concatenate([
            out[L-1] * (numpy.multiply.outer(alpha(L), 1-2*w).T - beta(L)).T,
            [delta(L) * out[L-1][L-1] * (u-v)],
            ])
            )

        if L > 1:
            out[-1][:L-1] -= (out[L-2].T * gamma(L)).T
            out[-1][-1] -= epsilon(L) * out[L-2][L-2] * (u+v)**2

    return out
