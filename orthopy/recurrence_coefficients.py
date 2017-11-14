# -*- coding: utf-8 -*-
#
from __future__ import division

import numpy
import sympy


def chebyshev1(n):
    p0 = 1
    a = n * [1]
    b = n * [0]
    c = n * [sympy.Rational(1, 4)]
    c[0] = sympy.pi
    c[1] = sympy.Rational(1, 2)
    return p0, a, b, c


def chebyshev2(n):
    p0 = 1
    a = n * [1]
    b = n * [0]
    c = n * [sympy.Rational(1, 4)]
    c[0] = sympy.pi / 2
    return p0, a, b, c


def laguerre(n):
    return laguerre_generalized(n, 0)


def laguerre_generalized(n, alpha):
    p0 = 1
    a = n * [1]
    b = [(2*k+1+alpha) for k in range(n)]
    c = [k*(k+alpha) for k in range(n)]
    c[0] = sympy.gamma(alpha+1)
    return p0, a, b, c


def hermite(n):
    p0 = 1
    a = n * [1]
    b = n * [0]
    c = [sympy.Rational(k, 2) for k in range(n)]
    c[0] = sympy.sqrt(sympy.pi)
    return p0, a, b, c


def legendre(n, normalization='monic'):
    return jacobi(n, 0, 0, normalization=normalization)


def jacobi(n, alpha, beta, normalization='monic'):
    '''Generate the recurrence coefficients alpha_k, beta_k

    P_{k+1}(x) = (a_k x - b_k)*P_{k}(x) - c_k P_{k-1}(x)

    for the Jacobi polynomials which are orthogonal on [-1, 1]
    with respect to the weight w(x)=[(1-x)^alpha]*[(1+x)^beta].

    Adapted from the MATLAB code by Dirk Laurie and Walter Gautschi
    http://www.cs.purdue.edu/archives/2002/wxg/codes/r_jacobi.m
    and from Greg van Winckel's
    https://github.com/gregvw/orthopoly-quadrature/blob/master/rec_jacobi.pyx
    '''
    if normalization == 'monic':
        p0 = 1

        if n == 0:
            return p0, [], [], []

        b0 = sympy.Rational(beta-alpha, alpha+beta+2)

        # c[0] is not used in the actual recurrence, but is often defined
        # as the integral of the weight function of the domain, i.e.,
        # ```
        # int_{-1}^{+1} (1-x)^a * (1+x)^b dx =
        #     2^(a+b+1) * Gamma(a+1) * Gamma(b+1) / Gamma(a+b+2).
        # ```
        c0 = sympy.Rational(
            2**(alpha+beta+1) * sympy.gamma(alpha+1) * sympy.gamma(beta+1),
            sympy.gamma(alpha+beta+2)
            )

        if n == 1:
            return p0, [1], [b0], [c0]

        a = numpy.ones(n, dtype=int)

        N = range(1, n)

        nab = [2*nn + alpha + beta for nn in N]

        b = [b0] + [
            sympy.Rational(beta**2 - alpha**2, val * (val + 2)) for val in nab
            ]

        N = N[1:]
        nab = nab[1:]
        C1 = sympy.Rational(
            4 * (alpha+1) * (beta+1),
            (alpha+beta+2)**2 * (alpha+beta+3)
            )
        C = [
            sympy.Rational(
                4 * (nn+alpha) * (nn+beta) * nn * (nn+alpha+beta),
                val**2 * (val+1) * (val-1)
                )
            for nn, val in zip(N, nab)
            ]
        c = [c0, C1] + C

        out = p0, a, numpy.array(b), numpy.array(c)
    elif normalization == 'p(1)=(n+a over n)' \
            or (alpha == 0 and normalization == 'p(1)=1'):
        p0 = 1
        if n == 0:
            return p0, numpy.array([]), numpy.array([]), numpy.array([])

        a = [
            sympy.Rational(
                (2*N+alpha+beta-1) * (2*N+alpha+beta),
                2*N * (N+alpha+beta)
                )
            for N in range(1, n+1)
            ]

        b = [
            sympy.Rational(beta-alpha, 2) if N == 1 else
            sympy.Rational(
                (beta**2 - alpha**2) * (2*N+alpha+beta-1),
                2*N * (N+alpha+beta) * (2*N+alpha+beta-2)
                )
            for N in range(1, n+1)
            ]

        c0 = sympy.Rational(
            2**(alpha+beta+1) * sympy.gamma(alpha+1) * sympy.gamma(beta+1),
            sympy.gamma(alpha+beta+2)
            )
        c = [c0] + [
            sympy.Rational(
                2 * (N+alpha-1) * (N+beta-1) * (2*N+alpha+beta),
                2*N * (N+alpha+beta) * (2*N+alpha+beta-2)
                )
            for N in range(2, n+1)
            ]

        out = p0, numpy.array(a), numpy.array(b), numpy.array(c)
    else:
        assert normalization == '||p**2||=1', \
            'Unknown normalization \'{}\'.'.format(normalization)

        p0 = sympy.sqrt(sympy.Rational(
            sympy.gamma(alpha+beta+2),
            2**(alpha+beta+1) * sympy.gamma(alpha+1) * sympy.gamma(beta+1)
            ))

        if n == 0:
            return p0, [], [], []

        a = [
            sympy.Rational(2*N+alpha+beta, 2) * sympy.sqrt(sympy.Rational(
                    (2*N+alpha+beta-1) * (2*N+alpha+beta+1),
                    N * (N+alpha) * (N+beta) * (N+alpha+beta)
                    ))
            for N in range(1, n+1)
            ]

        b0 = sympy.Rational(beta-alpha, 2) \
            * sympy.sqrt(
                sympy.Rational(
                    (alpha+beta+3) * (alpha+beta+1),
                    (1+alpha) * (1+beta) * (alpha+beta+1)
                    ))
        b = [b0] + [
            sympy.Rational(
                beta**2 - alpha**2,
                2 * (2*N+alpha+beta-2)
                ) * sympy.sqrt(sympy.Rational(
                    (2*N+alpha+beta+1) * (2*N+alpha+beta-1),
                    N * (N+alpha) * (N+beta) * (N+alpha+beta)
                    ))
            for N in range(2, n+1)
            ]

        c0 = sympy.Rational(
            2**(alpha+beta+1) * sympy.gamma(alpha+1) * sympy.gamma(beta+1),
            sympy.gamma(alpha+beta+2)
            )
        c = [c0] + [
            sympy.Rational(2*N+alpha+beta, 2*N+alpha+beta-2)
            * sympy.sqrt(sympy.Rational(
                (N-1) * (N+alpha-1) * (N+beta-1) * (N+alpha+beta-1)
                * (2*N+alpha+beta+1),
                N * (N+alpha) * (N+beta) * (N+alpha+beta)
                * (2*N+alpha+beta-3)
                ))
            for N in range(2, n+1)
            ]

        out = p0, numpy.array(a), numpy.array(b), numpy.array(c)

    return out
