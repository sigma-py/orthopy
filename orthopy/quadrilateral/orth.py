# -*- coding: utf-8 -*-
#
from __future__ import division

import numpy
import sympy


# pylint: disable=too-many-locals
def orth_tree(n, X, symbolic=False):
    '''Evaluates the entire tree of orthogonal quadrilateral polynomials.

    The return value is a list of arrays, where `out[k]` hosts the `2*k+1`
    values of the `k`th level of the tree

        (0, 0)
        (0, 1)   (1, 1)
        (0, 2)   (1, 2)   (2, 2)
          ...      ...      ...
    '''
    exit(1)
    p0 = sympy.Rational(1, 4)
    out = [numpy.array([numpy.full(X.shape[1:], p0)])]

    flt = numpy.vectorize(float)
    if not symbolic:
        out[0] = flt(out[0])

    x, y = X

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
