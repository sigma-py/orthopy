# -*- coding: utf-8 -*-
#
from __future__ import division, print_function

import orthopy
import sympy


def test_jacobi():
    # alpha, beta = orthopy.recurrence_coefficients.jacobi(5, 0, 0)
    alpha, beta = orthopy.recurrence_coefficients.legendre(0)
    x = sympy.Rational(1, 2)
    val = orthopy.tools.evaluate_orthogonal_polynomial(alpha, beta, x)
    assert val == 1
    val = orthopy.tools.evaluate_orthogonal_polynomial(alpha, beta, 1)
    assert val == 1

    alpha, beta = orthopy.recurrence_coefficients.legendre(1)
    x = sympy.Rational(1, 2)
    val = orthopy.tools.evaluate_orthogonal_polynomial(alpha, beta, x)
    assert val == sympy.Rational(1, 2)
    val = orthopy.tools.evaluate_orthogonal_polynomial(alpha, beta, 1)
    assert val == 1

    alpha, beta = orthopy.recurrence_coefficients.legendre(5)
    # Not normalized! (normalized: 23/256)
    x = sympy.Rational(1, 2)
    val = orthopy.tools.evaluate_orthogonal_polynomial(alpha, beta, x)
    assert val == sympy.Rational(23, 2016)
    # Not normalized! (normalized: 1)
    val = orthopy.tools.evaluate_orthogonal_polynomial(alpha, beta, 1)
    assert val == sympy.Rational(8, 63)
    return


if __name__ == '__main__':
    test_jacobi()
