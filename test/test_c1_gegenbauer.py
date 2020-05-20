import numpy
import pytest
from sympy import Rational

import orthopy
from helpers import get_nth


# simple smoke test for gegenbauer
@pytest.mark.parametrize(
    "n, y",
    [
        (0, [1, 1, 1]),
        (1, [0, Rational(1, 2), 1]),
        (2, [-Rational(1, 2), -Rational(1, 4), Rational(1, 2)]),
        (3, [0, -Rational(1, 4), Rational(1, 4)]),
        (4, [Rational(1, 8), -Rational(1, 16), Rational(1, 8)]),
        (5, [0, Rational(1, 32), Rational(1, 16)]),
    ],
)
def test_gegenbauer_chebyshev1_monic(n, y):
    x = numpy.array([0, Rational(1, 2), 1])
    lmbda = -Rational(1, 2)

    # Test evaluation of one value
    y0 = get_nth(orthopy.c1.gegenbauer.Iterator(x[0], "monic", lmbda, symbolic=True), n)
    assert y0 == y[0]

    # Test evaluation of multiple values
    val = get_nth(orthopy.c1.gegenbauer.Iterator(x, "monic", lmbda, symbolic=True), n)
    assert all(val == y)
