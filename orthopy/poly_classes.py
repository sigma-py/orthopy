# -*- coding: utf-8 -*-
#
# pylint: disable=too-few-public-methods

import sympy


def legendre(k, x):
    '''Legendre polynomials scaled such that the leading term has coefficient
    1.
    '''
    return (
        sympy.Rational(2**k * sympy.factorial(k)**2, sympy.factorial(2*k))
        * sympy.polys.orthopolys.legendre_poly(k, x)
        )
