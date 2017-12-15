# -*- coding: utf-8 -*-
#
from __future__ import division

import numpy

from ..line.recurrence_coefficients import legendre


# pylint: disable=too-many-locals
def tree(n, X, symbolic=False):
    '''Evaluates the entire tree of orthogonal quadrilateral polynomials.

    The return value is a list of arrays, where `out[k]` hosts the `2*k+1`
    values of the `k`th level of the tree

        (0, 0)
        (0, 1)   (1, 1)
        (0, 2)   (1, 2)   (2, 2)
         ...      ...      ...
    '''
    p0, a, b, c = legendre(n+1, 'normal', symbolic=symbolic)

    p0 **= 2
    out = [numpy.array([numpy.full(X.shape[1:], p0)])]
    for L in range(1, n+1):
        alpha = numpy.array([a[L-1-m] for m in range(L)])
        beta = numpy.array([b[L-1-m] for m in range(L)])

        # The order of X and a is important here. If X is int and a is
        # sympy, then the product will be sympy for X*a and float for a*X.
        out.append(numpy.concatenate([
            out[L-1] * (numpy.multiply.outer(alpha, X[0]).T - beta).T,
            [out[L-1][-1] * (X[1] * a[L-1] - b[L-1])],
            ])
            )

        if L > 1:
            gamma = numpy.array([c[L-1-m] for m in range(L-1)])
            out[-1][:L-1] -= (out[L-2].T * gamma).T
            out[-1][-1] -= out[L-2][L-2] * c[L-1]  # okay

    return out
