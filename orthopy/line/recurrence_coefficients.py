# -*- coding: utf-8 -*-
#
from __future__ import division

import numpy
import scipy.special
import sympy


def chebyshev1(n, standardization, symbolic=False):
    frac = sympy.Rational if symbolic else lambda x, y: x/y
    return jacobi(
        n, -frac(1, 2), -frac(1, 2), standardization,
        symbolic=symbolic
        )


def chebyshev2(n, standardization, symbolic=False):
    frac = sympy.Rational if symbolic else lambda x, y: x/y
    return jacobi(
        n, +frac(1, 2), +frac(1, 2), standardization,
        symbolic=symbolic
        )


def laguerre(n, symbolic=False):
    return laguerre_generalized(n, 0, symbolic=symbolic)


def laguerre_generalized(n, alpha, symbolic=False):
    gamma = (
        sympy.gamma if symbolic else
        lambda x: scipy.special.gamma(float(x))
        )
    p0 = 1
    a = n * [1]
    b = [(2*k+1+alpha) for k in range(n)]
    c = [k*(k+alpha) for k in range(n)]
    c[0] = gamma(alpha+1)
    return p0, a, b, c


def hermite(n, standardization, symbolic=False):
    frac = sympy.Rational if symbolic else lambda x, y: x/y
    sqrt = sympy.sqrt if symbolic else numpy.sqrt
    pi = sympy.pi if symbolic else numpy.pi

    # Check <https://en.wikipedia.org/wiki/Hermite_polynomials> for the
    # different standardizations.
    if standardization in ['probabilist', 'monic']:
        p0 = 1
        a = numpy.ones(n, dtype=int)
        b = numpy.zeros(n, dtype=int)
        c = [k for k in range(n)]
        c[0] = numpy.nan
    elif standardization == 'physicist':
        p0 = 1
        a = numpy.full(n, 2, dtype=int)
        b = numpy.zeros(n, dtype=int)
        c = [2*k for k in range(n)]
        c[0] = sqrt(pi)  # only used for custom scheme
    else:
        assert standardization == 'normal', \
            'Unknown standardization \'{}\'.'.format(standardization)
        p0 = 1 / sqrt(sqrt(pi))
        a = [sqrt(frac(2, k+1)) for k in range(n)]
        b = numpy.zeros(n, dtype=int)
        c = [sqrt(frac(k, k+1)) for k in range(n)]
        c[0] = numpy.nan

    return p0, a, b, c


def legendre(n, standardization, symbolic=False):
    return jacobi(n, 0, 0, standardization, symbolic=symbolic)


def jacobi(n, alpha, beta, standardization, symbolic=False):
    '''Generate the recurrence coefficients a_k, b_k, c_k in

    P_{k+1}(x) = (a_k x - b_k)*P_{k}(x) - c_k P_{k-1}(x)

    for the Jacobi polynomials which are orthogonal on [-1, 1]
    with respect to the weight w(x)=[(1-x)^alpha]*[(1+x)^beta]; see
    <https://en.wikipedia.org/wiki/Jacobi_polynomials#Recurrence_relations>.
    '''
    gamma = (
        sympy.gamma if symbolic else
        lambda x: scipy.special.gamma(float(x))
        )
    frac = (
        # <https://github.com/sympy/sympy/pull/13670>
        lambda x, y: (
            sympy.Rational(x, y)
            if all([isinstance(val, int) for val in [x, y]])
            else x/y
            )
        if symbolic else
        lambda x, y: x/y
        )
    sqrt = sympy.sqrt if symbolic else numpy.sqrt

    int_1 = (
        2**(alpha+beta+1) * gamma(alpha+1) * gamma(beta+1)
        / gamma(alpha+beta+2)
        )

    if standardization == 'monic':
        p0 = 1

        a = numpy.ones(n, dtype=int)

        # work around bug <https://github.com/sympy/sympy/issues/13618>
        if isinstance(alpha, numpy.int64):
            alpha = int(alpha)
        if isinstance(beta, numpy.int64):
            beta = int(beta)

        b = [
            frac(beta-alpha, alpha+beta+2) if N == 0 else
            frac(
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
            frac(
                4 * (1+alpha) * (1+beta),
                (2+alpha+beta)**2 * (3+alpha+beta)
                ) if N == 1 else
            frac(
                4 * (N+alpha) * (N+beta) * N * (N+alpha+beta),
                (2*N+alpha+beta)**2 * (2*N+alpha+beta+1) * (2*N+alpha+beta-1)
                )
            for N in range(n)
            ]

    elif standardization == 'p(1)=(n+alpha over n)' \
            or (alpha == 0 and standardization == 'p(1)=1'):
        p0 = 1

        # work around bug <https://github.com/sympy/sympy/issues/13618>
        if isinstance(alpha, numpy.int64):
            alpha = int(alpha)
        if isinstance(beta, numpy.int64):
            beta = int(beta)

        # Treat N==0 separately to avoid division by 0 for alpha=beta=-1/2.
        a = [
            frac(alpha+beta+2, 2) if N == 0 else
            frac(
                (2*N+alpha+beta+1) * (2*N+alpha+beta+2),
                2*(N+1) * (N+alpha+beta+1)
                )
            for N in range(n)
            ]

        b = [
            frac(beta-alpha, 2) if N == 0 else
            frac(
                (beta**2 - alpha**2) * (2*N+alpha+beta+1),
                2*(N+1) * (N+alpha+beta+1) * (2*N+alpha+beta)
                )
            for N in range(n)
            ]

        c = [
            int_1 if N == 0 else
            frac(
                (N+alpha) * (N+beta) * (2*N+alpha+beta+2),
                (N+1) * (N+alpha+beta+1) * (2*N+alpha+beta)
                )
            for N in range(n)
            ]

    else:
        assert standardization == 'normal', \
            'Unknown standardization \'{}\'.'.format(standardization)

        p0 = sqrt(1 / int_1)

        # Treat N==0 separately to avoid division by 0 for alpha=beta=-1/2.
        a = [
            frac(alpha+beta+2, 2) * sqrt(frac(
                    alpha+beta+3,
                    (alpha+1) * (beta+1)
                    )) if N == 0 else
            frac(2*N+alpha+beta+2, 2) * sqrt(frac(
                    (2*N+alpha+beta+1) * (2*N+alpha+beta+3),
                    (N+1) * (N+alpha+1) * (N+beta+1) * (N+alpha+beta+1)
                    ))
            for N in range(n)
            ]

        b = [(
                frac(beta-alpha, 2) if N == 0 else
                frac(beta**2 - alpha**2, 2 * (2*N+alpha+beta))
             ) * sqrt(frac(
                    (2*N+alpha+beta+3) * (2*N+alpha+beta+1),
                    (N+1) * (N+alpha+1) * (N+beta+1) * (N+alpha+beta+1)
                    ))
             for N in range(n)
             ]

        c = [
            int_1 if N == 0 else
            frac(4+alpha+beta, 2+alpha+beta)
            * sqrt(frac(
                (1+alpha) * (1+beta) * (5+alpha+beta),
                2 * (2+alpha) * (2+beta) * (2+alpha+beta)
                )) if N == 1 else
            frac(2*N+alpha+beta+2, 2*N+alpha+beta)
            * sqrt(frac(
                N * (N+alpha) * (N+beta) * (N+alpha+beta)
                * (2*N+alpha+beta+3),
                (N+1) * (N+alpha+1) * (N+beta+1) * (N+alpha+beta+1)
                * (2*N+alpha+beta-1)
                ))
            for N in range(n)
            ]

    return p0, numpy.array(a), numpy.array(b), numpy.array(c)
