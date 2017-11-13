# -*- coding: utf-8 -*-
#
from __future__ import division

import sympy

from . import recurrence_coefficients
from . import tools


def legendre(n, x, mode='sympy', normalization='monic'):
    '''Evaluate Legendre polynomials.
    '''
    if normalization == 'p(1)=1':
        normalization = 'p(1)=(n+a over n)'
    return jacobi(n, 0, 0, x, mode=mode, normalization=normalization)


def jacobi(n, a, b, x, mode='sympy', normalization='monic'):
    '''Evaluate Jacobi polynomials.
    '''

    # P_n = c * (x + alpha) * P_{n-1} - beta * P_{n-2}
    alpha, beta, c = recurrence_coefficients.jacobi(
            n, a, b,
            normalization=normalization,
            mode=mode
            )
    out = tools.evaluate_orthogonal_polynomial(x, alpha, beta, c)

    # if normalization == 'monic':
    #     pass
    # elif normalization == 'p(1)=(n+a over n)' or (a == 0 and 'p(1)=1'):
    #     pass
    #     # alpha = sympy.Rational(
    #     #     2**n * sympy.factorial(n) * sympy.gamma(n + a + b + 1),
    #     #     sympy.gamma(2*n + a + b + 1)
    #     #     )
    #     # out = out / alpha
    # else:
    #     assert normalization == '||p**2||=1', \
    #         'Unknown normalization \'{}\'.'.format(normalization)
    #     alpha, beta, c = recurrence_coefficients.jacobi(n, a, b, mode=mode)
    #     out = tools.evaluate_orthogonal_polynomial(x, alpha, beta, c)
    #     alpha = sympy.Rational(
    #         2**n * sympy.factorial(n) * sympy.gamma(n + a + b + 1),
    #         sympy.gamma(2*n + a + b + 1)
    #         )
    #     alpha *= sympy.sqrt(
    #         sympy.Rational(2**(a+b+1), 2*n+a+b+1)
    #         * sympy.gamma(n+a+1) * sympy.gamma(n+b+1) / sympy.gamma(n+a+b+1)
    #         / sympy.factorial(n)
    #         )
    #     out = out / alpha

    return out
