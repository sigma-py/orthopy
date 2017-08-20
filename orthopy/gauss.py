# -*- coding: utf-8 -*-
#
import warnings

import numpy

from .helpers import _gauss


class Gauss(object):
    '''Given moments

    mu_k = int_a^b omega(x) x^k dx,  k = {0, 1,...,2N}

    (with omega being a nonnegative weight function), this class creates the
    Gauss scheme corresponding to the above integral. It uses the mechanism
    from

    Gene H. Golub and John H. Welsch,
    Calculation of Gauss Quadrature Rules,
    Mathematics of Computation,
    Vol. 23, No. 106 (Apr., 1969), pp. 221-230+s1-s10,
    <https://dx.doi.org/10.2307/2004418>,
    <https://pdfs.semanticscholar.org/c715/119d5464f614fd8ec590b732ccfea53e72c4.pdf>.
    '''
    def __init__(self, n, moments):
        self.degree = 2*n-1

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

        err_alpha, err_beta = check_coefficients(moments, alpha, beta)

        max_error = max(numpy.max(err_alpha), numpy.max(err_beta))
        if max_error > 1.0e-12:
            warnings.warn(
                'The sanity test shows an error of {:e}. '.format(max_error) +
                'Handle with care!'
                )

        self.points, self.weights = _gauss(alpha, beta)
        return


def check_coefficients(moments, alpha, beta):
    '''
    In his article

    How and how not to check Gaussian quadrature formulae,
    BIT Numerical Mathematics,
    June 1983, Volume 23, Issue 2, pp 209–216,

    Walter Gautschi suggests a method for checking if a Gauss quadrature rule
    is sane. This method implements test #3 for the article.
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
