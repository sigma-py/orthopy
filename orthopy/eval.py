# -*- coding: utf-8 -*-
#
from __future__ import division

import numpy
import sympy

from . import recurrence_coefficients


def legendre(n, x, mode='sympy', normalization='monic'):
    '''Evaluate Legendre polynomials.
    '''
    if normalization == 'p(1)=1':
        normalization = 'p(1)=(n+a over n)'
    return jacobi(n, 0, 0, x, mode=mode, normalization=normalization)


def jacobi(n, a, b, x, mode='sympy', normalization='monic'):
    '''Evaluate Jacobi polynomials.
    '''
    alpha, beta = recurrence_coefficients.jacobi(n, a, b, mode=mode)
    out = evaluate_orthogonal_polynomial(alpha, beta, x)

    if normalization == 'monic':
        pass
    elif normalization == 'p(1)=(n+a over n)':
        alpha = sympy.Rational(
            2**n * sympy.factorial(n) * sympy.gamma(n + a + b + 1),
            sympy.gamma(2*n + a + b + 1)
            )
        out = out / alpha
    else:
        assert normalization == '||p**2||=1', \
            'Unknown normalization \'{}\'.'.format(normalization)
        alpha = sympy.Rational(
            2**n * sympy.factorial(n) * sympy.gamma(n + a + b + 1),
            sympy.gamma(2*n + a + b + 1)
            )
        alpha *= sympy.sqrt(
            sympy.Rational(2**(a+b+1), 2*n+a+b+1)
            * sympy.gamma(n+a+1) * sympy.gamma(n+b+1) / sympy.gamma(n+a+b+1)
            / sympy.factorial(n)
            )
        out = out / alpha

    return out


def evaluate_orthogonal_polynomial(alpha, beta, t):
    '''Evaluate the orthogonal polynomial defined by its recurrence coefficients
    alpha, beta at the point(s) t.
    '''
    try:
        vals1 = numpy.zeros(t.shape, dtype=int)
    except AttributeError:
        vals1 = 0

    try:
        vals2 = numpy.ones(t.shape, dtype=int)
    except AttributeError:
        vals2 = 1

    for alpha_k, beta_k in zip(alpha, beta):
        vals0 = vals1
        vals1 = vals2
        vals2 = (t - alpha_k) * vals1 - beta_k * vals0
    return vals2
