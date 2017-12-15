# -*- coding: utf-8 -*-
#
from __future__ import division

import numpy
import sympy

from ..line.recurrence_coefficients import legendre


# pylint: disable=too-many-locals
def orth_tree(n, X, symbolic=True):
    '''Evaluates the entire tree of orthogonal quadrilateral polynomials.

    The return value is a list of arrays, where `out[k]` hosts the `2*k+1`
    values of the `k`th level of the tree

        (0, 0)
        (0, 1)   (1, 1)
        (0, 2)   (1, 2)   (2, 2)
         ...      ...      ...
    '''
    frac = sympy.Rational if symbolic else lambda x, y: x/y
    sqrt = sympy.sqrt if symbolic else numpy.sqrt

    p0, a, b, c = legendre(n+1, 'normal', symbolic=symbolic)

    p0 *= sqrt(frac(1, 2))
    out = [numpy.array([numpy.full(X.shape[1:], p0)])]
    for L in range(1, n+1):
        out.append(numpy.concatenate([
            # The order of X and a is important here. If X is int and a is
            # sympy, then the product will be sympy for X*a and float for a*X.
            out[L-1] * (X[0] * a[L] - b[L]),
            [out[L-1][-1] * (X[1] * a[L] - b[L])],
            ])
            )

        if L > 1:
            out[-1][:-1] -= out[L-2] * c[L]
            out[-1][-1] -= out[L-2][L-2] * c[L]

    return out
