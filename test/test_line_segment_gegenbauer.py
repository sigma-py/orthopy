import numpy
import pytest
from sympy import Rational

import orthopy


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

    out = orthopy.line_segment.recurrence_coefficients.gegenbauer(
        n, -Rational(1, 2), "monic", symbolic=True
    )

    # Test evaluation of one value
    y0 = orthopy.tools.line_evaluate(x[0], *out)
    assert y0 == y[0]

    # Test evaluation of multiple values
    val = orthopy.tools.line_evaluate(x, *out)
    assert all(val == y)
    return
