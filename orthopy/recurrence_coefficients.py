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
    '''Generate the recurrence coefficients a_k, b_k, c_k in

    P_{k+1}(x) = (a_k x - b_k)*P_{k}(x) - c_k P_{k-1}(x)

    for the Jacobi polynomials which are orthogonal on [-1, 1]
    with respect to the weight w(x)=[(1-x)^alpha]*[(1+x)^beta]; see
    <https://en.wikipedia.org/wiki/Jacobi_polynomials#Recurrence_relations>.
    '''
    if normalization == 'monic':
        p0 = 1

        a = numpy.ones(n, dtype=int)

        b = [
            sympy.Rational(beta-alpha, alpha+beta+2) if N == 0 else
            sympy.Rational(
                beta**2 - alpha**2,
                (2*N+alpha+beta) * (2*N+alpha+beta+2)
                )
            for N in range(n)
            ]

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
        c = [
            c0 if N == 0 else
            sympy.Rational(
                4 * (N+alpha) * (N+beta) * N * (N+alpha+beta),
                (2*N+alpha+beta)**2 * (2*N+alpha+beta+1) * (2*N+alpha+beta-1)
                )
            for N in range(n)
            ]

    elif normalization == 'p(1)=(n+a over n)' \
            or (alpha == 0 and normalization == 'p(1)=1'):
        p0 = 1

        a = [
            sympy.Rational(
                (2*N+alpha+beta+1) * (2*N+alpha+beta+2),
                2*(N+1) * (N+alpha+beta+1)
                )
            for N in range(n)
            ]

        b = [
            sympy.Rational(beta-alpha, 2) if N == 0 else
            sympy.Rational(
                (beta**2 - alpha**2) * (2*N+alpha+beta+1),
                2*(N+1) * (N+alpha+beta+1) * (2*N+alpha+beta)
                )
            for N in range(n)
            ]

        c0 = sympy.Rational(
            2**(alpha+beta+1) * sympy.gamma(alpha+1) * sympy.gamma(beta+1),
            sympy.gamma(alpha+beta+2)
            )
        c = [
            c0 if N == 0 else
            sympy.Rational(
                2 * (N+alpha) * (N+beta) * (2*N+alpha+beta+2),
                2*(N+1) * (N+alpha+beta+1) * (2*N+alpha+beta)
                )
            for N in range(n)
            ]

    else:
        assert normalization == '||p**2||=1', \
            'Unknown normalization \'{}\'.'.format(normalization)

        p0 = sympy.sqrt(sympy.Rational(
            sympy.gamma(alpha+beta+2),
            2**(alpha+beta+1) * sympy.gamma(alpha+1) * sympy.gamma(beta+1)
            ))

        a = [
            sympy.Rational(2*N+alpha+beta+2, 2) * sympy.sqrt(sympy.Rational(
                    (2*N+alpha+beta+1) * (2*N+alpha+beta+3),
                    (N+1) * (N+alpha+1) * (N+beta+1) * (N+alpha+beta+1)
                    ))
            for N in range(n)
            ]

        b = [(
                sympy.Rational(beta-alpha, 2) if N == 0 else
                sympy.Rational(beta**2 - alpha**2, 2 * (2*N+alpha+beta))
            ) * sympy.sqrt(sympy.Rational(
                (2*N+alpha+beta+3) * (2*N+alpha+beta+1),
                (N+1) * (N+alpha+1) * (N+beta+1) * (N+alpha+beta+1)
                ))
            for N in range(n)
            ]

        c0 = sympy.Rational(
            2**(alpha+beta+1) * sympy.gamma(alpha+1) * sympy.gamma(beta+1),
            sympy.gamma(alpha+beta+2)
            )
        c = [
            c0 if N == 0 else
            sympy.Rational(2*N+alpha+beta+2, 2*N+alpha+beta)
            * sympy.sqrt(sympy.Rational(
                N * (N+alpha) * (N+beta) * (N+alpha+beta)
                * (2*N+alpha+beta+3),
                (N+1) * (N+alpha+1) * (N+beta+1) * (N+alpha+beta+1)
                * (2*N+alpha+beta-1)
                ))
            for N in range(n)
            ]

    return p0, numpy.array(a), numpy.array(b), numpy.array(c)
