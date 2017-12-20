# -*- coding: utf-8 -*-
#
from __future__ import division

import numpy
import scipy.special

from ..line.recurrence_coefficients import legendre


# pylint: disable=too-many-locals
def tree(n, X, symbolic=False):
    '''Evaluates the entire tree of orthogonal polynomials for the n-cube

    The computation is organized such that tree returns a list of arrays, L={0,
    ..., dim}, where each level corresponds to the polynomial degree L.
    Further, each level is organized like a discrete (dim-1)-dimensional
    simplex. Let's demonstrate this for 3D:

    L = 1:
         (0, 0, 0)

    L = 2:
         (1, 0, 0)
         (0, 1, 0) (0, 0, 1)

    L = 3:
         (2, 0, 0)
         (1, 1, 0) (1, 0, 1)
         (0, 2, 0) (0, 1, 1) (0, 0, 2)

    The main insight here that makes computation for n dimensions easy is that
    the next level is composed by:

       * Taking the whole previous level and adding +1 to the first entry.
       * Taking the last row of the previous level and adding +1 to the second
         entry.
       * Taking the last entry of the last row of the previous and adding +1 to
         the third entry.

    In the same manner this can be repeated for `dim` dimensions.
    '''
    p0, a, b, c = legendre(n+1, 'normal', symbolic=symbolic)

    dim = X.shape[0]

    p0n = p0 ** dim
    out = []

    level = numpy.array([numpy.ones(X.shape[1:], dtype=int) * p0n])
    out.append(level)

    # TODO use a simpler binom implementation
    for L in range(1, n+1):
        # The order of X and a is important here. If X is int and a is
        # sympy, then the product will be sympy for X*a and float for a*X.
        level = []
        print('L={}'.format(L))
        for i in range(dim):
            r = 0
            m1 = int(scipy.special.binom(L+dim-i-2, dim-i-1))
            dat1 = out[L-1][-m1:]
            print('dat1', dat1)
            m2 = int(scipy.special.binom(L+dim-i-2, dim-i))
            dat2 = out[L-2][-m2:]
            # TODO ugly, replace
            num = L if i < dim-1 else 1
            print('    {}, {}, {}'.format(i, m1, m2))
            print(level)
            for k in range(num):
                m = int(scipy.special.binom(k+dim-2, dim-2))
                print('        k = {}, m = {}'.format(k, m))
                val = dat1[r:r+m] * (a[L-k-1] * X[i] - b[L-k-1])
                if L-k-1 > 0:
                    print('        ({}, {})'.format(r, r+m))
                    val -= dat2[r:r+m] * c[L-k-1]
                r += m
                level.append(val)

        out.append(numpy.concatenate(level))

    for o in out:
        print(o)
    return out
