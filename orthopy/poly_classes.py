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
