# -*- coding: utf-8 -*-
#
from __future__ import division

import numpy
import sympy


# pylint: disable=too-many-locals
def tree(n, X, symbolic=False):
    '''Evaluates the entire tree of orthogonal polynomials on the unit disk.

    The return value is a list of arrays, where `out[k]` hosts the `2*k+1`
    values of the `k`th level of the tree

        (0, 0)
        (0, 1)   (1, 1)
        (0, 2)   (1, 2)   (2, 2)
          ...      ...      ...
    '''
    frac = sympy.Rational if symbolic else lambda x, y: x/y
    sqrt = sympy.sqrt if symbolic else numpy.sqrt
    pi = sympy.pi if symbolic else numpy.pi

    mu = frac(1, 2)

    p0 = 1 / sqrt(pi)

    def alpha(n, k):
        return sqrt(frac(
            (n-k) * (n+mu+frac(1, 2)),
            (n+k+2*mu+1) * (n+mu-frac(1, 2))
            )) / (n-k) * (n+mu-frac(1, 2))

    def alpha1(n):
        return 2 * numpy.array([alpha(n, k) for k in range(n)])

    def alpha2(n):
        return 2 * alpha(n, 0)

    def gamma(n, k):
        return sqrt(frac(
            (n-k) * (n+mu+frac(1, 2)),
            (n+k+2*mu+1) * (n+mu-frac(1, 2))
            )) / (n-k) * (n+k+2*mu-1)

    def gamma1(n):
        return numpy.array([gamma(n, k) for k in range(n-1)])

    def gamma2(n):
        return gamma(n, 0)

    out = [numpy.array([0 * X[0] + p0])]

    for L in range(1, n+1):
        out.append(numpy.concatenate([
            out[L-1] * numpy.multiply.outer(alpha1(L), X[0]),
            [out[L-1][L-1] * X[1] * alpha2(L)],
            ])
            )

        if L > 1:
            out[-1][:L-1] -= (out[L-2].T * gamma1(L)).T
            out[-1][-1] -= out[L-2][L-2] * gamma2(L)

    return out
