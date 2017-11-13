# -*- coding: utf-8 -*-
#
from __future__ import division

import math

import numpy
import sympy


def chebyshev1(n):
    alpha = n * [0]
    beta = n * [sympy.Rational(1, 4)]
    beta[0] = sympy.pi
    beta[1] = sympy.Rational(1, 2)
    return alpha, beta


def chebyshev2(n):
    alpha = n * [0]
    beta = n * [sympy.Rational(1, 4)]
    beta[0] = sympy.pi / 2
    return alpha, beta


def laguerre(n):
    return laguerre_generalized(n, 0)


def laguerre_generalized(n, a):
    alpha = [(2*k+1+a) for k in range(n)]
    beta = [k*(k+a) for k in range(n)]
    beta[0] = sympy.gamma(a+1)
    return alpha, beta


def hermite(n):
    alpha = n * [0]
    beta = [sympy.Rational(k, 2) for k in range(n)]
    beta[0] = sympy.sqrt(sympy.pi)
    return alpha, beta


def legendre(n, mode='sympy'):
    return jacobi(n, 0, 0, mode=mode)


def jacobi(n, a, b, normalization='monic', mode='sympy'):
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
        alpha, beta, gamma = \
            _jacobi_sympy(n, a, b, normalization=normalization)
    else:
        assert mode == 'numpy'
        alpha, beta, gamma = \
            _jacobi_numpy(n, a, b, normalization=normalization)
    return alpha, beta, gamma


def _jacobi_sympy(n, a, b, normalization):

    if normalization == 'monic':
        alpha0 = sympy.Rational(b-a, a+b+2)

        # beta[0] is not used in the actual recurrence, but is often defined
        # as the integral of the weight function of the domain, i.e.,
        # ```
        # int_{-1}^{+1} (1-x)^a * (1+x)^b dx =
        #     2^(a+b+1) * Gamma(a+1) * Gamma(b+1) / Gamma(a+b+2).
        # ```
        # orthopy assumed beta[0] to contain the value of P_0.
        beta0 = 1

        if n == 0:
            return [], [beta0], []
        elif n == 1:
            return [alpha0], [beta0], [1]

        N = range(1, n)

        nab = [2*nn + a + b for nn in N]

        alpha = [alpha0] + [
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
        beta = [beta0, B1] + B

        c = numpy.array(n * [1])

        out = numpy.array(alpha), numpy.array(beta), c
    elif normalization == 'p(1)=(n+a over n)' \
            or (a == 0 and normalization == 'p(1)=1'):
        beta0 = 1
        if n == 0:
            return [], [beta0], []

        alpha0 = sympy.Rational(b-a, a+b+2)
        alpha = [alpha0]
        if n > 0:
            N = 1
            alpha.append(
                sympy.Rational(b**2 - a**2, (2*N+a+b) * (2*N+a+b+2))
                )
        alpha += [
            sympy.Rational(
                (b**2 - a**2) * (2*N+a+b-1),
                2*N * (N+a+b) * (2*N+a+b-2)
                )
            for N in range(2, n)
            ]

        beta = [beta0] + [
            sympy.Rational(
                2 * (N+a-1) * (N+b-1) * (2*N+a+b),
                2*N * (N+a+b) * (2*N+a+b-2)
                )
            for N in range(2, n+1)
            ]

        c0 = 1
        c = [c0] + [
            sympy.Rational(
                (2*N+a+b-1) * (2*N+a+b),
                2*N * (N+a+b)
                )
            for N in range(2, n+1)
            ]

        out = numpy.array(alpha), numpy.array(beta), numpy.array(c)
    else:
        assert normalization == '||p**2||=1', \
            'Unknown normalization \'{}\'.'.format(normalization)

        beta0 = sympy.sqrt(sympy.Rational(1, 2))
        if n == 0:
            return [], [beta0], []

        alpha0 = sympy.Rational(b-a, a+b+2) * sympy.sqrt(
                sympy.Rational(
                    (a+b+3) * (a+b+1),
                    (1+a) * (1+b) * (a+b+1)
                    ))
        alpha = [alpha0]
        if n > 0:
            N = 1
            alpha.append(
                sympy.Rational(b**2 - a**2, (2*N+a+b) * (2*N+a+b+2))
                )
        alpha += [
            sympy.Rational(
                (b**2 - a**2) * (2*N+a+b-1),
                2*N * (N+a+b) * (2*N+a+b-2)
                )
            for N in range(2, n)
            ]

        beta = [beta0] + [
            sympy.Rational(
                2 * (N+a-1) * (N+b-1) * (2*N+a+b),
                2*N * (N+a+b) * (2*N+a+b-2)
                ) * sympy.sqrt(sympy.Rational(
                    (2*N+a+b+1) * (N+a+b) * (N+a+b-1) * N * (N-1),
                    (N+a) * (N+a-1) * (N+b) * (N+b-1) * (2*N+a+b-3)
                    ))
            for N in range(2, n+1)
            ]

        c0 = sympy.sqrt(sympy.Rational(
                (a+b+3) * (a+b+1),
                (1+a) * (1+b) * (a+b+1),
                ))
        c = [c0] + [
            sympy.Rational(
                (2*N+a+b-1) * (2*N+a+b),
                2*N * (N+a+b)
                ) * sympy.sqrt(sympy.Rational(
                    (2*N+a+b+1) * (N+a+b) * N,
                    (N+a) * (N+b) * (2*N+a+b-1)
                    ))
            for N in range(2, n+1)
            ]

        out = numpy.array(alpha), numpy.array(beta), numpy.array(c)

    return out


def _jacobi_numpy(n, a, b, normalization):
    assert normalization == 'monic'

    if n == 0:
        return numpy.array([]), numpy.array([]), numpy.array([])

    mu = 2.0**(a+b+1.0) \
        * numpy.exp(
            math.lgamma(a+1.0) + math.lgamma(b+1.0) - math.lgamma(a+b+2.0)
            )
    nu = (b-a) / (a+b+2.0)

    if n == 1:
        return numpy.array([nu]), numpy.array([mu])

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

    gamma = numpy.ones(n)
    return alpha, beta, gamma
