# -*- coding: utf-8 -*-
#
import sympy

from .main import gauss_from_coefficients
from . import recurrence_coefficients


def legendre(n, decimal_places):
    alpha, beta = recurrence_coefficients.legendre(n, mode='sympy')
    return gauss_from_coefficients(
            alpha, beta, mode='mpmath',
            decimal_places=decimal_places
            )


def jacobi(a, b, n, decimal_places):
    alpha, beta = recurrence_coefficients.jacobi(n, a, b, mode='sympy')
    return gauss_from_coefficients(
            alpha, beta, mode='mpmath',
            decimal_places=decimal_places
            )


def chebyshev1(n, decimal_places):
    # There are explicit representations, too, but for the sake of consistency
    # go for the recurrence coefficients approach here.
    alpha, beta = recurrence_coefficients.chebyshev1(n)
    return gauss_from_coefficients(
            alpha, beta, mode='mpmath',
            decimal_places=decimal_places
            )


def chebyshev2(n, decimal_places):
    # There are explicit representations, too, but for the sake of consistency
    # go for the recurrence coefficients approach here.
    alpha, beta = recurrence_coefficients.chebyshev2(n)
    beta[0] = sympy.N(beta[0], decimal_places)
    return gauss_from_coefficients(
            alpha, beta, mode='mpmath',
            decimal_places=decimal_places
            )


def laguerre(n, decimal_places):
    alpha, beta = recurrence_coefficients.laguerre(n)
    return gauss_from_coefficients(
            alpha, beta, mode='mpmath',
            decimal_places=decimal_places
            )
