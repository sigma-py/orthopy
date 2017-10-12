# -*- coding: utf-8 -*-
#
from __future__ import division, print_function

import orthopy
import pytest
import sympy


@pytest.mark.parametrize('n, val1, val2', [
    (0, 1, 1),
    (1, sympy.Rational(1, 2), 1),
    # Not normalized! (normalized: 23/256, 1)
    (5, sympy.Rational(23, 2016), sympy.Rational(8, 63)),
    ]
    )
def test_legendre(n, val1, val2):
    alpha, beta = orthopy.recurrence_coefficients.legendre(n)
    x = sympy.Rational(1, 2)
    val = orthopy.tools.evaluate_orthogonal_polynomial(alpha, beta, x)
    assert val == val1
    val = orthopy.tools.evaluate_orthogonal_polynomial(alpha, beta, 1)
    assert val == val2
    return


if __name__ == '__main__':
    test_legendre(0, 1, 1)
