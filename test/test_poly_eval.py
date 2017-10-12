# -*- coding: utf-8 -*-
#
import numpy
import orthopy
import pytest
from sympy import Rational


@pytest.mark.parametrize('n, val0, val1, val2', [
    (0, 1, 1, 1),
    (1, 0, Rational(1, 2), 1),
    (2, -Rational(1, 3), -Rational(1, 12), Rational(2, 3)),
    (3, 0, -Rational(7, 40), Rational(2, 5)),
    (4, Rational(3, 35), -Rational(37, 560), Rational(8, 35)),
    # Not normalized! (normalized: 23/256, 1)
    (5, 0, Rational(23, 2016), Rational(8, 63)),
    ]
    )
def test_legendre(n, val0, val1, val2):
    alpha, beta = orthopy.recurrence_coefficients.legendre(n)

    # Test evaluation of one value
    val = orthopy.tools.evaluate_orthogonal_polynomial(alpha, beta, 0)
    assert val == val0

    # Test evaluation of multiple values
    x = numpy.array([Rational(1, 2), 1])
    val = orthopy.tools.evaluate_orthogonal_polynomial(alpha, beta, x)
    assert all(val == numpy.array([val1, val2]))
    return


if __name__ == '__main__':
    test_legendre(0, 1, 1, 1)
