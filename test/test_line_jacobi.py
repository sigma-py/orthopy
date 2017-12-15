# -*- coding: utf-8 -*-
#
import numpy
import pytest
from sympy import Rational, sqrt

import orthopy


@pytest.mark.parametrize('n, y', [
    (0, [1, 1, 1]),
    (1, [Rational(1, 7), Rational(9, 14), Rational(8, 7)]),
    (2, [-Rational(1, 9), Rational(1, 4), Rational(10, 9)]),
    (3, [-Rational(1, 33), Rational(7, 264), Rational(32, 33)]),
    (4, [Rational(3, 143), -Rational(81, 2288), Rational(112, 143)]),
    (5, [Rational(1, 143), -Rational(111, 4576), Rational(256, 429)]),
    ]
    )
def test_jacobi_monic(n, y):
    x = numpy.array([0, Rational(1, 2), 1])

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
    (1, [Rational(1, 2), Rational(9, 4), 4]),
    (2, [-1, Rational(9, 4), 10]),
    (3, [-Rational(5, 8), Rational(35, 64), 20]),
    (4, [Rational(15, 16), -Rational(405, 256), 35]),
    (5, [Rational(21, 32), -Rational(2331, 1024), 56]),
    ]
    )
def test_jacobi_p11(n, y):
    x = numpy.array([0, Rational(1, 2), 1])

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
    x = numpy.array([0, Rational(1, 2), 1])

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
