# -*- coding: utf-8 -*-
#
import numpy
import pytest
from sympy import S, sqrt

import orthopy


@pytest.mark.parametrize('n, y', [
    (0, [1, 1, 1]),
    (1, [S(1)/7, S(9)/14, S(8)/7]),
    (2, [-S(1)/9, S(1)/4, S(10)/9]),
    (3, [-S(1)/33, S(7)/264, S(32)/33]),
    (4, [S(3)/143, -S(81)/2288, S(112)/143]),
    (5, [S(1)/143, -S(111)/4576, S(256)/429]),
    ]
    )
def test_jacobi_monic(n, y):
    x = numpy.array([0, S(1)/2, 1])

    out = orthopy.line.recurrence_coefficients.jacobi(
            n, alpha=3, beta=2, standardization='monic', symbolic=True
            )

    y2 = orthopy.line.evaluate_orthogonal_polynomial(x[2], *out)
    assert y2 == y[2]

    val = orthopy.line.evaluate_orthogonal_polynomial(x, *out)
    assert all(val == y)
    return


@pytest.mark.parametrize('n, y', [
    (0, [1, 1, 1]),
    (1, [S(1)/2, S(9)/4, 4]),
    (2, [-1, S(9)/4, 10]),
    (3, [-S(5)/8, S(35)/64, 20]),
    (4, [S(15)/16, -S(405)/256, 35]),
    (5, [S(21)/32, -S(2331)/1024, 56]),
    ]
    )
def test_jacobi_p11(n, y):
    x = numpy.array([0, S(1)/2, 1])

    out = orthopy.line.recurrence_coefficients.jacobi(
            n, alpha=3, beta=2, standardization='p(1)=(n+alpha over n)',
            symbolic=True
            )

    y2 = orthopy.line.evaluate_orthogonal_polynomial(x[2], *out)
    assert y2 == y[2]

    val = orthopy.line.evaluate_orthogonal_polynomial(x, *out)
    assert all(val == y)
    return


@pytest.mark.parametrize('n, y', [
    (0, [sqrt(15)/4, sqrt(15)/4, sqrt(15)/4]),
    (1, [sqrt(10)/8, 9*sqrt(10)/16, sqrt(10)]),
    (2, [-sqrt(35)/8, 9*sqrt(35)/32, 5*sqrt(35)/4]),
    (3, [-sqrt(210)/32, 7*sqrt(210)/256, sqrt(210)]),
    (4, [3*sqrt(210)/64, -81*sqrt(210)/1024, 7*sqrt(210)/4]),
    (5, [3*sqrt(105)/64, -333*sqrt(105)/2048, 4*sqrt(105)]),
    ]
    )
def test_jacobi_normal(n, y):
    x = numpy.array([0, S(1)/2, 1])

    out = orthopy.line.recurrence_coefficients.jacobi(
            n, alpha=3, beta=2, standardization='normal', symbolic=True
            )

    y2 = orthopy.line.evaluate_orthogonal_polynomial(x[2], *out)
    assert y2 == y[2]

    val = orthopy.line.evaluate_orthogonal_polynomial(x, *out)
    assert all(val == y)
    return


if __name__ == '__main__':
    test_jacobi_monic(0, [1, 1, 1])
