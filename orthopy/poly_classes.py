# -*- coding: utf-8 -*-
#
# pylint: disable=too-few-public-methods

import sympy


def legendre(k, x, monic=True):
    '''Legendre polynomials, optionally (default=True) scaled such that the
    leading term has coefficient 1.
    '''
    coeff = (
        sympy.Rational(2**k * sympy.factorial(k)**2, sympy.factorial(2*k))
        if monic
        else 1
        )
    return coeff * sympy.polys.orthopolys.legendre_poly(k, x)


def jacobi(k, a, b, x, monic=True):
    '''Jacobi polynomials, optionally (default=True) scaled such that the
    leading term has coefficient 1.
    '''
    coeff = (
        sympy.Rational(
            2**k * sympy.factorial(k) * sympy.gamma(k + a + b + 1),
            sympy.gamma(2**k + a + b + 1)
            )
        if monic
        else 1
        )
    return coeff * sympy.polys.orthopolys.jacobi_poly(k, a, b, x)


def chebyshev1(k, x, monic=True):
    '''Chebyshev polynomials of the first kind, optionally (default=True)
    scaled such that the leading term has coefficient 1.
    '''
    return jacobi(
            k, -sympy.Rational(1, 2), -sympy.Rational(1, 2), x, monic=monic
            )


def chebyshev2(k, x, monic=True):
    '''Chebyshev polynomials of the second kind, optionally (default=True)
    scaled such that the leading term has coefficient 1.
    '''
    return jacobi(
            k, sympy.Rational(1, 2), sympy.Rational(1, 2), x, monic=monic
            )


def hermite(k, x, monic=True):
    '''Hermite polynomials, optionally (default=True) scaled such that the
    leading term has coefficient 1.
    '''
    coeff = sympy.Rational(1, 2**k) if monic else 1
    return coeff * sympy.polys.orthopolys.hermite_poly(k, x)


def laguerre(k, x, monic=True):
    '''Laguerre polynomials, optionally (default=True) scaled such that the
    leading term has coefficient 1.
    '''
    coeff = sympy.Rational((-1)**k, sympy.factorial(k)) if monic else 1
    return coeff * sympy.polys.orthopolys.laguerre_poly(k, x)
