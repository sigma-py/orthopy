# -*- coding: utf-8 -*-
#
import mpmath
import orthopy


def test_tridiag_eigen():
    # build 3x3 reference matrix
    n = 3
    A = mpmath.zeros(3, n)
    for i in range(n):
        A[i, i] = 2
    for i in range(n-1):
        A[i, i+1] = -1
        A[i+1, i] = -1
    vals0, vecs0 = mpmath.eigsy(A)

    d = +2 * mpmath.ones(3, 1)
    e = -1 * mpmath.ones(3, 1)

    vals1, vecs1 = orthopy.tridiag_eigen(d, e)

    tol = 1.0e-12
    assert mpmath.norm(vals1 - vals0, p=mpmath.inf) < tol
    assert mpmath.norm(vecs1 - vecs0, p=mpmath.inf) < tol
    return


if __name__ == '__main__':
    test_tridiag_eigen()
