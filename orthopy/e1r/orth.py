# -*- coding: utf-8 -*-
#
from __future__ import division

import scipy.special
import sympy

from ..tools import line_tree


def tree(n, X, symbolic=False):
    args = recurrence_coefficients(n, symbolic=symbolic)
    return line_tree(X, *args)


def recurrence_coefficients(n, symbolic=False):
    '''Recurrence coefficients for Laguerre polynomials.
    '''
    return recurrence_coefficients_generalized(n, 0, symbolic=symbolic)


def recurrence_coefficients_generalized(n, alpha, symbolic=False):
    '''Recurrence coefficients for generalized Laguerre polynomials.

        vals2 = vals1 * (t*a_k - b_k) - vals0 * c_k
    '''
    gamma = (
        sympy.gamma if symbolic else
        lambda x: scipy.special.gamma(float(x))
        )
    p0 = 1
    a = n * [1]
    b = [(2*k+1+alpha) for k in range(n)]
    c = [k*(k+alpha) for k in range(n)]
    c[0] = gamma(alpha+1)
    return p0, a, b, c
