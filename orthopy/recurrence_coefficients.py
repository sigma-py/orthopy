# -*- coding: utf-8 -*-
#
from __future__ import division

import numpy
import sympy


def chebyshev1(n, standardization):
    return jacobi(
        n, -sympy.Rational(1, 2), -sympy.Rational(1, 2), standardization
        )


def chebyshev2(n, standardization):
    return jacobi(
        n, +sympy.Rational(1, 2), +sympy.Rational(1, 2), standardization
        )


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


def legendre(n, standardization):
    return jacobi(n, 0, 0, standardization)


def jacobi(n, alpha, beta, standardization):
    '''Generate the recurrence coefficients a_k, b_k, c_k in

    P_{k+1}(x) = (a_k x - b_k)*P_{k}(x) - c_k P_{k-1}(x)

    for the Jacobi polynomials which are orthogonal on [-1, 1]
    with respect to the weight w(x)=[(1-x)^alpha]*[(1+x)^beta]; see
    <https://en.wikipedia.org/wiki/Jacobi_polynomials#Recurrence_relations>.
    '''
    int_1 = (
        2**(alpha+beta+1) * sympy.gamma(alpha+1) * sympy.gamma(beta+1)
        / sympy.gamma(alpha+beta+2)
        )

    if standardization == 'monic':
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
        # Note also that we have the treat the case N==1 separately to
        # division by 0 for alpha=beta=-1/2.
        c = [
            int_1 if N == 0 else
            sympy.Rational(
                4 * (1+alpha) * (1+beta),
                (2+alpha+beta)**2 * (3+alpha+beta)
                ) if N == 1 else
            sympy.Rational(
                4 * (N+alpha) * (N+beta) * N * (N+alpha+beta),
                (2*N+alpha+beta)**2 * (2*N+alpha+beta+1) * (2*N+alpha+beta-1)
                )
            for N in range(n)
            ]

    elif standardization == 'p(1)=(n+alpha over n)' \
            or (alpha == 0 and standardization == 'p(1)=1'):
        p0 = 1

        # Treat N==0 separately to avoid division by 0 for alpha=beta=-1/2.
        a = [
            sympy.Rational(alpha+beta+2, 2) if N == 0 else
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

        c = [
            int_1 if N == 0 else
            sympy.Rational(
                2 * (N+alpha) * (N+beta) * (2*N+alpha+beta+2),
                2*(N+1) * (N+alpha+beta+1) * (2*N+alpha+beta)
                )
            for N in range(n)
            ]

    else:
        assert standardization == '||p**2||=1', \
            'Unknown standardization \'{}\'.'.format(standardization)

        p0 = sympy.sqrt(1 / int_1)

        # Treat N==0 separately to avoid division by 0 for alpha=beta=-1/2.
        a = [
            sympy.Rational(alpha+beta+2, 2) * sympy.sqrt(sympy.Rational(
                    alpha+beta+3,
                    (alpha+1) * (beta+1)
                    )) if N == 0 else
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

        c = [
            int_1 if N == 0 else
            sympy.Rational(4+alpha+beta, 2+alpha+beta)
            * sympy.sqrt(sympy.Rational(
                (1+alpha) * (1+beta) * (5+alpha+beta),
                2 * (2+alpha) * (2+beta) * (2+alpha+beta)
                )) if N == 1 else
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
