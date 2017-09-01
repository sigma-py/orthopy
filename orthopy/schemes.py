# -*- coding: utf-8 -*-
#

from .main import gauss_from_coefficients
from .rc import (
    recurrence_coefficients_jacobi,
    recurrence_coefficients_chebyshev1,
    recurrence_coefficients_chebyshev2,
    )


def legendre(n, decimal_places):
    alpha, beta = recurrence_coefficients_jacobi(n, 0, 0, mode='sympy')
    return gauss_from_coefficients(
            alpha, beta, mode='mpmath',
            decimal_places=decimal_places
            )


def jacobi(a, b, n, decimal_places):
    alpha, beta = recurrence_coefficients_jacobi(n, a, b, mode='sympy')
    return gauss_from_coefficients(
            alpha, beta, mode='mpmath',
            decimal_places=decimal_places
            )


def chebyshev1(n, decimal_places):
    # There are explicit representations, too, but for the sake of consistency
    # go for the recurrence coefficients approach here.
    alpha, beta = recurrence_coefficients_chebyshev1(n, mode='sympy')
    return gauss_from_coefficients(
            alpha, beta, mode='mpmath',
            decimal_places=decimal_places
            )


def chebyshev2(n, decimal_places):
    # There are explicit representations, too, but for the sake of consistency
    # go for the recurrence coefficients approach here.
    alpha, beta = recurrence_coefficients_chebyshev2(n, mode='sympy')
    return gauss_from_coefficients(
            alpha, beta, mode='mpmath',
            decimal_places=decimal_places
            )
