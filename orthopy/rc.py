# -*- coding: utf-8 -*-
#
from __future__ import division

import math

import numpy
import sympy


def recurrence_coefficients_legendre(n, mode='sympy'):
    return recurrence_coefficients_jacobi(n, 0, 0, mode=mode)


def recurrence_coefficients_jacobi(n, a, b, mode='sympy'):
    '''Generate the recurrence coefficients alpha_k, beta_k

    P_{k+1}(x) = (x-alpha_k)*P_{k}(x) - beta_k P_{k-1}(x)

    for the Jacobi polynomials which are orthogonal on [-1, 1]
    with respect to the weight w(x)=[(1-x)^a]*[(1+x)^b].

    Adapted from the MATLAB code by Dirk Laurie and Walter Gautschi
    http://www.cs.purdue.edu/archives/2002/wxg/codes/r_jacobi.m
    and from Greg van Winckel's
    https://github.com/gregvw/orthopoly-quadrature/blob/master/rec_jacobi.pyx
    '''
    assert a > -1.0 or b > -1.0

    if mode == 'sympy':
        if n == 0:
            return [], []

        assert n > 1

        mu = 2**(a+b+1) * sympy.Rational(
            sympy.gamma(a+1) * sympy.gamma(b+1), sympy.gamma(a+b+2)
            )
        nu = sympy.Rational(b-a, a+b+2)

        if n == 1:
            return nu, mu

        N = range(1, n)

        nab = [2*nn + a + b for nn in N]

        alpha = [nu] + [
            sympy.Rational(b**2 - a**2, val * (val + 2)) for val in nab
            ]

        N = N[1:]
        nab = nab[1:]
        B1 = sympy.Rational(4 * (a+1) * (b+1), (a+b+2)**2 * (a+b+3))
        B = [
            sympy.Rational(
                4 * (nn+a) * (nn+b) * nn * (nn+a+b),
                val**2 * (val+1) * (val-1)
                )
            for nn, val in zip(N, nab)
            ]
        beta = [mu, B1] + B

        alpha = numpy.array(alpha)
        beta = numpy.array(beta)
    else:
        assert mode == 'numpy'
        if n == 0:
            return numpy.array([]), numpy.array([])

        assert n > 1

        mu = 2.0**(a+b+1.0) \
            * numpy.exp(
                math.lgamma(a+1.0) + math.lgamma(b+1.0) - math.lgamma(a+b+2.0)
                )
        nu = (b-a) / (a+b+2.0)

        if n == 1:
            return nu, mu

        N = numpy.arange(1, n)

        nab = 2.0*N + a + b
        alpha = numpy.hstack([nu, (b**2 - a**2) / (nab * (nab + 2.0))])
        N = N[1:]
        nab = nab[1:]
        B1 = 4.0 * (a+1.0) * (b+1.0) / ((a+b+2.0)**2.0 * (a+b+3.0))
        B = (
            4.0 * (N+a) * (N+b) * N * (N+a+b)
            / (nab**2.0 * (nab+1.0) * (nab-1.0))
            )
        beta = numpy.hstack((mu, B1, B))
    return alpha, beta