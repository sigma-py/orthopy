# -*- coding: utf-8 -*-
#
from __future__ import division

import numpy
import sympy

from ..line import tree as line_tree


def recurrence_coefficients(n, standardization, symbolic=False):
    frac = sympy.Rational if symbolic else lambda x, y: x/y
    sqrt = sympy.sqrt if symbolic else numpy.sqrt
    pi = sympy.pi if symbolic else numpy.pi

    # Check <https://en.wikipedia.org/wiki/Hermite_polynomials> for the
    # different standardizations.
    if standardization in ['probabilist', 'monic']:
        p0 = 1
        a = numpy.ones(n, dtype=int)
        b = numpy.zeros(n, dtype=int)
        c = [k for k in range(n)]
        c[0] = numpy.nan
    elif standardization == 'physicist':
        p0 = 1
        a = numpy.full(n, 2, dtype=int)
        b = numpy.zeros(n, dtype=int)
        c = [2*k for k in range(n)]
        c[0] = sqrt(pi)  # only used for custom scheme
    else:
        assert standardization == 'normal', \
            'Unknown standardization \'{}\'.'.format(standardization)
        p0 = 1 / sqrt(sqrt(pi))
        a = [sqrt(frac(2, k+1)) for k in range(n)]
        b = numpy.zeros(n, dtype=int)
        c = [sqrt(frac(k, k+1)) for k in range(n)]
        c[0] = numpy.nan

    return p0, a, b, c


def tree(n, X, symbolic=False):
    args = recurrence_coefficients(
            n, standardization='normal', symbolic=symbolic
            )
    return line_tree(X, *args)
