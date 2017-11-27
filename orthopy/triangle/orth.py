# -*- coding: utf-8 -*-
#
from __future__ import division

import numpy
import sympy


# pylint: disable=too-many-locals
def orth_tree(n, bary, standardization, symbolic=False):
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
    if standardization == '1':
        p0 = 1

        def alpha(n):
            return numpy.array([
                sympy.Rational(n*(2*n+1), (n-r) * (n+r+1))
                for r in range(n)
                ])

        def beta(n):
            return numpy.array([
                sympy.Rational(n * (2*r+1)**2, (n-r) * (n+r+1) * (2*n-1))
                for r in range(n)
                ])

        def gamma(n):
            return numpy.array([sympy.Rational(
                (n-r-1) * (n+r) * (2*n+1), (n-r) * (n+r+1) * (2*n-1)
                ) for r in range(n-1)
                ])

        def delta(n):
            return sympy.Rational(2*n-1, n)

        def epsilon(n):
            return sympy.Rational(n-1, n)

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
        p0 = sympy.sqrt(2)

        def alpha(n):
            return numpy.array([
                sympy.Rational(2*n+1, (n-r) * (n+r+1)) * sympy.sqrt((n+1)*n)
                for r in range(n)
                ])

        def beta(n):
            return numpy.array([
                sympy.Rational((2*r+1)**2, (n-r) * (n+r+1) * (2*n-1))
                * sympy.sqrt((n+1)*n)
                for r in range(n)
                ])

        def gamma(n):
            return numpy.array([sympy.Rational(
                (n-r-1) * (n+r) * (2*n+1),
                (n-r) * (n+r+1) * (2*n-1)
                ) for r in range(n-1)
                ]) * sympy.sqrt(sympy.Rational(n+1, n-1))

        def delta(n):
            return sympy.sqrt(sympy.Rational((2*n+1) * (n+1) * (2*n-1), n**3))

        def epsilon(n):
            return sympy.sqrt(sympy.Rational(
                (2*n+1) * (n+1) * (n-1), (2*n-3) * n**2
                ))

    out = [numpy.array([numpy.full(bary.shape[1:], p0)])]

    flt = numpy.vectorize(float)
    if not symbolic:
        out[0] = flt(out[0])

    u, v, w = bary

    for L in range(1, n+1):
        a = alpha(L)
        b = beta(L)
        d = delta(L)

        if not symbolic:
            a = flt(a)
            b = flt(b)
            d = flt(d)

        out.append(numpy.concatenate([
            out[L-1] * (numpy.multiply.outer(a, 1-2*w).T - b).T,
            [out[L-1][L-1] * (u-v) * d],
            ])
            )

        if L > 1:
            c = gamma(L)
            e = epsilon(L)

            if not symbolic:
                c = flt(c)
                e = flt(e)

            out[-1][:L-1] -= (out[L-2].T * c).T
            out[-1][-1] -= out[L-2][L-2] * (u+v)**2 * e

    return out
