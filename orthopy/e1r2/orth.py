# -*- coding: utf-8 -*-
#
from __future__ import division

import numpy
import sympy

from ..tools import line_tree


def recurrence_coefficients(n, standardization, symbolic=False):
    S = numpy.vectorize(sympy.S) if symbolic else lambda x: x
    sqrt = numpy.vectorize(sympy.sqrt) if symbolic else numpy.sqrt
    pi = sympy.pi if symbolic else numpy.pi

    # Check <https://en.wikipedia.org/wiki/Hermite_polynomials> for the
    # different standardizations.
    N = numpy.arange(n)
    if standardization in ['probabilist', 'monic']:
        p0 = 1
        a = numpy.ones(n, dtype=int)
        b = numpy.zeros(n, dtype=int)
        c = N
        c[0] = sqrt(pi)  # only used for custom scheme
    elif standardization == 'physicist':
        p0 = 1
        a = numpy.full(n, 2, dtype=int)
        b = numpy.zeros(n, dtype=int)
        c = 2*N
        c[0] = sqrt(pi)  # only used for custom scheme
    else:
        assert standardization == 'normal', \
            'Unknown standardization \'{}\'.'.format(standardization)
        p0 = 1 / sqrt(sqrt(pi))
        a = sqrt(S(2) / (N+1))
        b = numpy.zeros(n, dtype=int)
        c = sqrt(S(N) / (N+1))
        c[0] = numpy.nan

    return p0, a, b, c


def tree(n, X, symbolic=False):
    args = recurrence_coefficients(
            n, standardization='normal', symbolic=symbolic
            )
    return line_tree(X, *args)
