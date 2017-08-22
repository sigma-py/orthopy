# -*- coding: utf-8 -*-
#
# pylint: disable=too-few-public-methods
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
import math

import numpy
# pylint: disable=no-name-in-module
from scipy.linalg.lapack import get_lapack_funcs


def gauss_from_coefficients(alpha, beta):
    '''Compute the Gauss nodes and weights from the recursion coefficients
    associated with a set of orthogonal polynomials. See [2] and
    <http://www.scientificpython.net/pyblog/radau-quadrature>.
    '''
    from scipy.linalg import eig_banded
    import scipy
    A = numpy.vstack((numpy.sqrt(beta), alpha))
    x, V = eig_banded(A, lower=False)
    w = beta[0]*scipy.real(scipy.power(V[0, :], 2))
    return x, w


def clenshaw(a, alpha, beta, t):
    '''Clenshaw's algorithm for evaluating

    S(t) = \\sum a_k P_k(alpha, beta)(t)

    where P_k(alpha, beta) is the kth orthogonal polynomial defined by the
    recursion coefficients alpha, beta.

    See <https://en.wikipedia.org/wiki/Clenshaw_algorithm> for details.
    '''
    n = len(alpha)
    assert len(beta) == n
    assert len(a) == n + 1

    try:
        b = numpy.empty((n+1,) + t.shape)
    except AttributeError:  # 'float' object has no attribute 'shape'
        b = numpy.empty(n+1)

    b[0] = 3.1415

    b[n] = a[n]
    b[n-1] = a[n-1] + (t - alpha[n-1]) * b[n]
    for k in range(n-2, 0, -1):
        b[k] = a[k] + (t - alpha[k]) * b[k+1] - beta[k+1] * b[k+2]

    phi0 = 1.0
    phi1 = t - alpha[0]

    return phi0 * a[0] + phi1 * b[1] - beta[1] * phi0 * b[2]


def evaluate_orthogonal_polynomial(alpha, beta, t):
    '''Evaluate the ortogonal polynomial defined by its recursion coefficients
    alpha, beta at the point(s) t.
    '''
    n = len(alpha)
    assert len(beta) == n

    try:
        vals = numpy.empty((n+1,) + t.shape)
    except AttributeError:  # 'float' object has no attribute 'shape'
        vals = numpy.empty(n+1)

    vals[0] = 1.0
    vals[1] = (t - alpha[0]) * vals[0]
    for k in range(1, n):
        vals[k+1] = (t - alpha[k]) * vals[k] - beta[k] * vals[k-1]
    return vals[-1]


def golub_welsch(moments):
    '''Given moments

    mu_k = int_a^b omega(x) x^k dx,  k = {0, 1,...,2N}

    (with omega being a nonnegative weight function), this method creates the
    recursion coefficients of the corresponding orthogonal polynomials, see
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

    alpha = numpy.zeros(n)
    alpha = q.copy()
    alpha[+1:] -= q[:-1]

    # TODO don't square here, but adapt _gauss to accept squared values at
    #      input
    beta = numpy.hstack([
        Rd[0], Rd[1:-1] / Rd[:-2]
        ])**2
    return alpha, beta


def chebyshev(moments):
    '''Given the first 2n moments `int t^k dt`, this method uses the Chebyshev
    algorithm (see, e.g., [2]) for computing the associated recursion
    coefficients.

    Ill-conditioned, see [2].
    '''
    m = len(moments)
    assert m % 2 == 0
    return chebyshev_modified(moments, numpy.zeros(m), numpy.zeros(m))


def chebyshev_modified(nu, a, b):
    '''Given the first 2n modified moments `nu_k = int p_k(t) dt`, where the
    p_k are orthogonal polynomials with recursion coefficients a, b, this
    method implements the modified Chebyshev algorithm (see, e.g., [2]) for
    computing the associated recursion coefficients.
    '''
    m = len(nu)
    assert m % 2 == 0

    n = m // 2

    alpha = numpy.empty(n)
    beta = numpy.empty(n)
    # Actually overkill. One could alternatively make sigma a list, and store
    # the shrinking rows there, only ever keeping the last two.
    sigma = numpy.empty((n, 2*n))

    k = 0
    sigma[k, k:2*n-k] = nu
    alpha[0] = a[0] + nu[1] / nu[0]
    beta[0] = nu[0]

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


def jacobi_recursion_coefficients(n, a, b):
    '''Generate the recursion coefficients alpha_k, beta_k

    P_{k+1}(x) = (x-alpha_k)*P_{k}(x) - beta_k P_{k-1}(x)

    for the Jacobi polynomials which are orthogonal on [-1, 1]
    with respect to the weight w(x)=[(1-x)^a]*[(1+x)^b].

    Adapted from the MATLAB code by Dirk Laurie and Walter Gautschi
    http://www.cs.purdue.edu/archives/2002/wxg/codes/r_jacobi.m
    and from Greg van Winckel's
    https://github.com/gregvw/orthopoly-quadrature/blob/master/rec_jacobi.pyx
    '''
    assert a > -1.0 or b > -1.0
    assert n >= 1

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
    B = 4.0 * (N+a) * (N+b) * N * (N+a+b) / (nab**2.0 * (nab+1.0) * (nab-1.0))
    beta = numpy.hstack((mu, B1, B))
    return alpha, beta


def coefficients_from_scheme(points, weights):
    '''Given the points and weights of a Gaussian quadrature rule, this method
    reconstructs the recursion coefficients alpha, beta as appearing in the
    tridiagonal Jacobi matrix tri(b, a, b).
    This is using "Method 2--orthogonal reduction" from (section 3.2 in [4]).
    The complexity is O(n^3); a faster method is suggested in 3.3 in [4].
    '''
    n = len(points)
    assert n == len(weights)

    A = numpy.zeros((n+1, n+1))

    # In sytrd, the _last_ row/column of Q are e, so put the values there.
    a00 = 1.0
    A[n, n] = a00
    k = numpy.arange(n)
    A[k, k] = points
    A[n, :-1] = numpy.sqrt(weights)
    A[:-1, n] = numpy.sqrt(weights)

    # Implemented in
    # <https://github.com/scipy/scipy/issues/7775>
    sytrd, sytrd_lwork = get_lapack_funcs(('sytrd', 'sytrd_lwork'))

    # query lwork (optional)
    lwork, info = sytrd_lwork(n+1)
    assert info == 0

    _, d, e, _, info = sytrd(A, lwork=lwork)
    assert info == 0

    return d[:-1][::-1], e[::-1]**2


def check_coefficients(moments, alpha, beta):
    '''In his article [3], Walter Gautschi suggests a method for checking if a
    Gauss quadrature rule is sane. This method implements test #3 for the
    article.
    '''
    n = len(alpha)
    assert len(beta) == n

    D = numpy.empty(n+1)
    Dp = numpy.empty(n+1)
    D[0] = 1.0
    D[1] = moments[0]
    Dp[0] = 0.0
    Dp[1] = moments[1]
    for k in range(2, n+1):
        A = numpy.array([moments[i:i+k] for i in range(k)])
        D[k] = numpy.linalg.det(A)
        #
        A[:, -1] = moments[k:2*k]
        Dp[k] = numpy.linalg.det(A)

    errors_alpha = numpy.zeros(n)
    errors_beta = numpy.zeros(n)

    errors_alpha[0] = abs(alpha[0] - (Dp[1]/D[1] - Dp[0]/D[0]))
    errors_beta[0] = abs(beta[0] - D[1])
    for k in range(1, n):
        errors_alpha[k] = abs(alpha[k] - (Dp[k+1]/D[k+1] - Dp[k]/D[k]))
        errors_beta[k] = abs(beta[k] - D[k+1]*D[k-1]/D[k]**2)

    return errors_alpha, errors_beta
