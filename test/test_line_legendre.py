# -*- coding: utf-8 -*-
#
import numpy
import orthopy
import pytest
from sympy import Rational, sqrt


@pytest.mark.parametrize('n, y', [
    (0, [1, 1, 1]),
    (1, [0, Rational(1, 2), 1]),
    (2, [-Rational(1, 3), -Rational(1, 12), Rational(2, 3)]),
    (3, [0, -Rational(7, 40), Rational(2, 5)]),
    (4, [Rational(3, 35), -Rational(37, 560), Rational(8, 35)]),
    (5, [0, Rational(23, 2016), Rational(8, 63)]),
    ]
    )
def test_legendre_monic(n, y):
    x = numpy.array([0, Rational(1, 2), 1])

    out = orthopy.line.recurrence_coefficients.legendre(
            n, 'monic', symbolic=True
            )

    # Test evaluation of one value
    y0 = orthopy.line.evaluate_orthogonal_polynomial(x[0], *out)
    assert y0 == y[0]

    # Test evaluation of multiple values
    val = orthopy.line.evaluate_orthogonal_polynomial(x, *out)
    assert all(val == y)
    return


@pytest.mark.parametrize('n, y', [
    (0, [1, 1, 1]),
    (1, [0, Rational(1, 2), 1]),
    (2, [-Rational(1, 2), -Rational(1, 8), 1]),
    (3, [0, -Rational(7, 16), 1]),
    (4, [Rational(3, 8), -Rational(37, 128), 1]),
    (5, [0, Rational(23, 256), 1]),
    ]
    )
def test_legendre_p11(n, y):
    x = numpy.array([0, Rational(1, 2), 1])

    out = orthopy.line.recurrence_coefficients.legendre(
            n, standardization='p(1)=1'
            )

    y0 = orthopy.line.evaluate_orthogonal_polynomial(x[0], *out)
    assert y0 == y[0]

    val = orthopy.line.evaluate_orthogonal_polynomial(x, *out)
    assert all(val == y)
    return


@pytest.mark.parametrize('n, y', [
    (0, [sqrt(Rational(1, 2)), sqrt(Rational(1, 2)), sqrt(Rational(1, 2))]),
    (1, [0, sqrt(Rational(3, 8)), sqrt(Rational(3, 2))]),
    (2, [
        -sqrt(Rational(5, 8)), -sqrt(Rational(5, 128)), sqrt(Rational(5, 2))
        ]),
    (3, [0, -sqrt(Rational(343, 512)), sqrt(Rational(7, 2))]),
    (4, [9 / sqrt(2) / 8, -111 / sqrt(2) / 128, 3/sqrt(2)]),
    (5, [0, sqrt(Rational(5819, 131072)), sqrt(Rational(11, 2))]),
    ]
    )
def test_legendre_normal(n, y):
    x = numpy.array([0, Rational(1, 2), 1])

    out = orthopy.line.recurrence_coefficients.legendre(
            n, standardization='normal', symbolic=True
            )

    y0 = orthopy.line.evaluate_orthogonal_polynomial(x[0], *out)
    assert y0 == y[0]

    val = orthopy.line.evaluate_orthogonal_polynomial(x, *out)
    assert all(val == y)
    return


if __name__ == '__main__':
    test_legendre_monic(0, [1, 1, 1])
