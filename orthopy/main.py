# -*- coding: utf-8 -*-
#
'''
[1] Gene H. Golub and John H. Welsch,
    Calculation of Gauss Quadrature Rules,
    Mathematics of Computation,
    Vol. 23, No. 106 (Apr., 1969), pp. 221-230+s1-s10,
    <https://dx.doi.org/10.2307/2004418>,
    <https://pdfs.semanticscholar.org/c715/119d5464f614fd8ec590b732ccfea53e72c4.pdf>.

[2] W. Gautschi,
    Algorithm 726: ORTHPOL–a package of routines for generating orthogonal
    polynomials and Gauss-type quadrature rules,
    ACM Transactions on Mathematical Software (TOMS),
    Volume 20, Issue 1, March 1994,
    Pages 21-62,
    <http://doi.org/10.1145/174603.174605>,
    <http://www.cs.purdue.edu/archives/2002/wxg/codes/gauss.m>,

[3] W. Gautschi,
    How and how not to check Gaussian quadrature formulae,
    BIT Numerical Mathematics,
    June 1983, Volume 23, Issue 2, pp 209–216,
    <https://doi.org/10.1007/BF02218441>.

[4] D. Boley and G.H. Golub,
    A survey of matrix inverse eigenvalue problems,
    Inverse Problems, 1987, Volume 3, Number 4,
    <https://doi.org/10.1088/0266-5611/3/4/010>.
'''
from __future__ import division

import numpy
import sympy


def golub_welsch(moments):
    '''Given moments

    mu_k = int_a^b omega(x) x^k dx,  k = {0, 1,...,2N}

    (with omega being a nonnegative weight function), this method creates the
    recurrence coefficients of the corresponding orthogonal polynomials, see
    section 4 ("Determining the Three Term Relationship from the Moments") in
    Golub-Welsch [1]. Numerically unstable, see [2].
    '''
    assert len(moments) % 2 == 1
    n = (len(moments) - 1) // 2

    M = numpy.array([[
        moments[i+j] for j in range(n+1)
        ] for i in range(n+1)])
    R = numpy.linalg.cholesky(M).T

    # (upper) diagonal
    Rd = R.diagonal()
    q = R.diagonal(1) / Rd[:-1]

    alpha = q.copy()
    alpha[+1:] -= q[:-1]

    # TODO don't square here, but adapt _gauss to accept square-rooted values
    #      as input
    beta = numpy.hstack([
        Rd[0], Rd[1:-1] / Rd[:-2]
        ])**2
    return alpha, beta


def stieltjes(w, a, b, n):
    t = sympy.Symbol('t')

    alpha = n * [None]
    beta = n * [None]
    mu = n * [None]
    pi = n * [None]

    k = 0
    pi[k] = 1
    mu[k] = sympy.integrate(pi[k]**2 * w(t), (t, a, b))
    alpha[k] = sympy.integrate(t * pi[k]**2 * w(t), (t, a, b)) / mu[k]
    beta[k] = mu[0]  # not used, by convection mu[0]

    k = 1
    pi[k] = (t - alpha[k-1]) * pi[k-1]
    mu[k] = sympy.integrate(pi[k]**2 * w(t), (t, a, b))
    alpha[k] = sympy.integrate(t * pi[k]**2 * w(t), (t, a, b)) / mu[k]
    beta[k] = mu[k] / mu[k-1]

    for k in range(2, n):
        pi[k] = (t - alpha[k-1]) * pi[k-1] - beta[k-1] * pi[k-2]
        mu[k] = sympy.integrate(pi[k]**2 * w(t), (t, a, b))
        alpha[k] = sympy.integrate(t * pi[k]**2 * w(t), (t, a, b)) / mu[k]
        beta[k] = mu[k] / mu[k-1]

    return alpha, beta


def chebyshev(moments):
    '''Given the first 2n moments `int t^k dt`, this method uses the Chebyshev
    algorithm (see, e.g., [2]) for computing the associated recurrence
    coefficients.

    WARNING: Ill-conditioned, see [2].
    '''
    m = len(moments)
    assert m % 2 == 0
    if isinstance(moments[0], tuple(sympy.core.all_classes)):
        dtype = sympy.Rational
    else:
        dtype = moments.dtype
    zeros = numpy.zeros(m, dtype=dtype)
    return chebyshev_modified(moments, zeros, zeros)


def chebyshev_modified(nu, a, b):
    '''Given the first 2n modified moments `nu_k = int p_k(t) dt`, where the
    p_k are orthogonal polynomials with recurrence coefficients a, b, this
    method implements the modified Chebyshev algorithm (see, e.g., [2]) for
    computing the associated recurrence coefficients.
    '''
    m = len(nu)
    assert m % 2 == 0

    n = m // 2

    alpha = numpy.empty(n, dtype=a.dtype)
    beta = numpy.empty(n, dtype=a.dtype)
    # Actually overkill. One could alternatively make sigma a list, and store
    # the shrinking rows there, only ever keeping the last two.
    sigma = numpy.empty((n, 2*n), dtype=a.dtype)

    if n > 0:
        k = 0
        sigma[k, k:2*n-k] = nu
        alpha[0] = a[0] + nu[1] / nu[0]
        beta[0] = nu[0]

    if n > 1:
        k = 1
        L = numpy.arange(k, 2*n-k)
        sigma[k, L] = (
            sigma[k-1, L+1]
            - (alpha[k-1] - a[L]) * sigma[k-1, L]
            + b[L] * sigma[k-1, L-1]
            )
        alpha[k] = (
            a[k]
            + sigma[k, k+1]/sigma[k, k]
            - sigma[k-1, k]/sigma[k-1, k-1]
            )
        beta[k] = sigma[k, k] / sigma[k-1, k-1]

    for k in range(2, n):
        L = numpy.arange(k, 2*n-k)
        sigma[k, L] = (
            sigma[k-1, L+1]
            - (alpha[k-1] - a[L]) * sigma[k-1, L]
            - beta[k-1] * sigma[k-2, L]
            + b[L] * sigma[k-1, L-1]
            )
        alpha[k] = (
            a[k]
            + sigma[k, k+1] / sigma[k, k]
            - sigma[k-1, k] / sigma[k-1, k-1]
            )
        beta[k] = sigma[k, k] / sigma[k-1, k-1]

    return alpha, beta


def compute_moments(w, a, b, n, polynomial_class=lambda k, x: x**k):
    '''Symbolically calculate the first n moments

      int_a^b w(x) P_k(x) dx

    where `P_k` is the `k`th polynomials of a specified class. The default
    settings are monomials, i.e., `P_k(x)=x^k`, but you can provide any
    function with the signature `p(k, x)`, e.g.,
    `sympy.polys.orthopolys.legendre_poly` scaled by the inverse of its leading
    coefficient `(2n)! / 2^n / (n!)^2`.
    '''
    x = sympy.Symbol('x')
    return numpy.array([
        sympy.integrate(w(x) * polynomial_class(k, x), (x, a, b))
        for k in range(n)
        ])
