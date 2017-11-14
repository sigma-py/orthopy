# -*- coding: utf-8 -*-
#
from __future__ import division

from . import recurrence_coefficients
from . import tools


def legendre(n, x, normalization='monic'):
    '''Evaluate Legendre polynomials.
    '''
    if normalization == 'p(1)=1':
        normalization = 'p(1)=(n+a over n)'
    return jacobi(n, 0, 0, x, normalization=normalization)


def jacobi(n, alpha, beta, x, normalization='monic'):
    '''Evaluate Jacobi polynomials.
    '''
    # P_n = (a*x - b) * P_{n-1} - c * P_{n-2}
    p0, a, b, c = recurrence_coefficients.jacobi(
            n, alpha, beta,
            normalization=normalization
            )
    return tools.evaluate_orthogonal_polynomial(x, p0, a, b, c)
