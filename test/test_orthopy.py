# -*- coding: utf-8 -*-
#
import numpy
import orthopy


def test_coefficients_from_moments():
    '''Test the custom Gauss generator with the weight function x**2.
    '''
    alpha = 2.0

    def integrate_exact(k):
        # \int_{-1}^{+1} |x^alpha| x^k
        return [
            0.0 if kk % 2 == 1 else 2.0/(alpha+kk+1)
            for kk in k
            ]

    # Get the moment corresponding to the weight function omega(x) =
    # x^alpha:
    #
    #                                     / 0 if k is odd,
    #    int_{-1}^{+1} |x^alpha| x^k dx ={
    #                                     \ 2/(alpha+k+1) if k is even.
    #
    n = 5
    k = numpy.arange(2*n+1)
    moments = (1.0 + (-1.0)**k) / (k + alpha + 1)
    alpha, beta = orthopy.coefficients_from_moments(n, moments)

    tol = 1.0e-14
    assert numpy.all(abs(alpha) < tol)
    assert abs(beta[0] - 2.0/3.0) < tol
    assert abs(beta[1] - 3.0/5.0) < tol
    assert abs(beta[2] - 4.0/35.0) < tol
    assert abs(beta[3] - 25.0/63.0) < tol
    assert abs(beta[4] - 16.0/99.0) < tol
    return


if __name__ == '__main__':
    test_coefficients_from_moments()
