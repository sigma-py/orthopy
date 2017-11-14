# -*- coding: utf-8 -*-
#
from __future__ import division

from . import recurrence_coefficients
from . import tools


def legendre(n, x, standardization='monic'):
    '''Evaluate Legendre polynomials.
    '''
    if standardization == 'p(1)=1':
        standardization = 'p(1)=(n+a over n)'
    return jacobi(n, 0, 0, x, standardization=standardization)


def jacobi(n, alpha, beta, x, standardization='monic'):
    '''Evaluate Jacobi polynomials.
    '''
    # P_n = (a*x - b) * P_{n-1} - c * P_{n-2}
    p0, a, b, c = recurrence_coefficients.jacobi(
            n, alpha, beta,
            standardization=standardization
            )
    return tools.evaluate_orthogonal_polynomial(x, p0, a, b, c)
