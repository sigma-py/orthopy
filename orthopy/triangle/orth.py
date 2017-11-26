# -*- coding: utf-8 -*-
#
from __future__ import division

import numpy


# pylint: disable=too-many-arguments
def orth_tree(n, bary, standardization):
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
        return _standardization_1(n, bary)

    assert standardization == 'normal'
    return _standardization_normal(n, bary)


def _standardization_1(n, bary):
    out = [numpy.full(bary.shape[1:], 1.0)]

    def alpha(n, r):
        return n*(2*n+1) / (n-r) / (n+r+1)

    def beta(n, r):
        return n * (2*r+1)**2 / (n-r) / (n+r+1) / (2*n-1)

    def gamma(n, r):
        return (n-r-1) * (n+r) * (2*n+1) / (n-r) / (n+r+1) / (2*n-1)

    u, v, w = bary

    if n > 0:
        L = 1
        out.append([
            (alpha(L, 0) * (1-2*w) - beta(L, 0)) * out[0][0],
            (2*L-1)/L * (u-v) * out[0][0],
            ])

    for L in range(2, n+1):
        out.append([
            + (alpha(L, r) * (1-2*w) - beta(L, r)) * out[L-1][r]
            - gamma(L, r) * out[L-2][r]
            for r in range(L-1)
            ] +
            [
            (alpha(L, L-1) * (1-2*w) - beta(L, L-1)) * out[L-1][L-1]
            ] +
            [
            + (2*L-1)/L * (u-v) * out[L-1][L-1]
            - (L-1)/L * (u+v)**2 * out[L-2][L-2]
            ]
            )
    return out


def _standardization_normal(n, bary):

    def alpha(n, r):
        return n*(2*n+1) / (n-r) / (n+r+1)

    def beta(n, r):
        return n * (2*r+1)**2 / (n-r) / (n+r+1) / (2*n-1)

    def gamma(n, r):
        return (n-r-1) * (n+r) * (2*n+1) / (n-r) / (n+r+1) / (2*n-1)

    def norm2(n, r):
        '''||P_{n,r}||^2 of the polynomials scaled as in the standardization-1
        case.
        '''
        # This probably has a much simpler representation. For reference, see
        # page 219 (and the reference to Gould, 1972) in
        #
        #  Farouki, Goodman, Sauer,
        #  Construction of orthogonal bases for polynomials in Bernstein form
        #  on triangular and simplex domains,
        #  Computer Aided Geometric Design 20 (2003) 209–230.
        #
        # The below expression is actually derived with the help of this
        # article, in particular the insight that
        #
        #   int_T P_{n, r}^2 =
        #       int_0^1 L_r^2(t) dt * int_0^1 q_{n,r}^2(w) (1-w)^(r+s+1) dw.
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
        # is always 1.
        return 1 / (2*r+1) / (2*n+2)

    out = [[numpy.full(bary.shape[1:], 1 / numpy.sqrt(norm2(0, 0)))]]

    u, v, w = bary

    if n > 0:
        out.append([
            (3*(1-2*w) - 1) / numpy.sqrt(2) * out[0][0],
            (u-v) * numpy.sqrt(6) * out[0][0],
            ])

    for L in range(2, n+1):
        out.append([
            + (alpha(L, r) * (1-2*w) - beta(L, r))
            * numpy.sqrt(norm2(L-1, r)/norm2(L, r)) * out[L-1][r]
            - gamma(L, r) * numpy.sqrt(norm2(L-2, r)/norm2(L, r)) * out[L-2][r]
            for r in range(L-1)
            ] +
            [
            (alpha(L, L-1) * (1-2*w) - beta(L, L-1))
            * numpy.sqrt((L+1) / L) * out[L-1][L-1]
            ] +
            [
            + 1/L * (u-v)
            * numpy.sqrt((2*L+1) * (2*L+2) * (2*L-1) / (2*L)) * out[L-1][L-1]
            - (L-1)/L * (u+v)**2
            * numpy.sqrt((2*L+1) * (2*L+2) / (2*L-3) / (2*L-2)) * out[L-2][L-2]
            ]
            )
    return out
