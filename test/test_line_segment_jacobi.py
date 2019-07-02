import numpy
import pytest
from sympy import S, sqrt

import orthopy


@pytest.mark.parametrize(
    "n, y",
    [
        (0, [1, 1, 1]),
        (1, [S(1) / 7, S(9) / 14, S(8) / 7]),
        (2, [-S(1) / 9, S(1) / 4, S(10) / 9]),
        (3, [-S(1) / 33, S(7) / 264, S(32) / 33]),
        (4, [S(3) / 143, -S(81) / 2288, S(112) / 143]),
        (5, [S(1) / 143, -S(111) / 4576, S(256) / 429]),
    ],
)
def test_jacobi_monic(n, y):
    x = numpy.array([0, S(1) / 2, 1])

    out = orthopy.line_segment.recurrence_coefficients.jacobi(
        n, alpha=3, beta=2, standardization="monic", symbolic=True
    )

    y2 = orthopy.tools.line_evaluate(x[2], *out)
    assert y2 == y[2]

    val = orthopy.tools.line_evaluate(x, *out)
    assert all(val == y)
    return


@pytest.mark.parametrize(
    "n, y",
    [
        (0, [1, 1, 1]),
        (1, [S(1) / 2, S(9) / 4, 4]),
        (2, [-1, S(9) / 4, 10]),
        (3, [-S(5) / 8, S(35) / 64, 20]),
        (4, [S(15) / 16, -S(405) / 256, 35]),
        (5, [S(21) / 32, -S(2331) / 1024, 56]),
    ],
)
def test_jacobi_p11(n, y):
    x = numpy.array([0, S(1) / 2, 1])

    out = orthopy.line_segment.recurrence_coefficients.jacobi(
        n, alpha=3, beta=2, standardization="p(1)=(n+alpha over n)", symbolic=True
    )

    y2 = orthopy.tools.line_evaluate(x[2], *out)
    assert y2 == y[2]

    val = orthopy.tools.line_evaluate(x, *out)
    assert all(val == y)
    return


@pytest.mark.parametrize(
    "n, y",
    [
        (0, [sqrt(15) / 4, sqrt(15) / 4, sqrt(15) / 4]),
        (1, [sqrt(10) / 8, 9 * sqrt(10) / 16, sqrt(10)]),
        (2, [-sqrt(35) / 8, 9 * sqrt(35) / 32, 5 * sqrt(35) / 4]),
        (3, [-sqrt(210) / 32, 7 * sqrt(210) / 256, sqrt(210)]),
        (4, [3 * sqrt(210) / 64, -81 * sqrt(210) / 1024, 7 * sqrt(210) / 4]),
        (5, [3 * sqrt(105) / 64, -333 * sqrt(105) / 2048, 4 * sqrt(105)]),
    ],
)
def test_jacobi_normal(n, y):
    x = numpy.array([0, S(1) / 2, 1])

    out = orthopy.line_segment.recurrence_coefficients.jacobi(
        n, alpha=3, beta=2, standardization="normal", symbolic=True
    )

    y2 = orthopy.tools.line_evaluate(x[2], *out)
    assert y2 == y[2]

    val = orthopy.tools.line_evaluate(x, *out)
    assert all(val == y)
    return


@pytest.mark.parametrize("dtype", [numpy.float, S])
def test_jacobi(dtype):
    n = 5
    if dtype == S:
        a = S(1)
        b = S(1)
        _, _, alpha, beta = orthopy.line_segment.recurrence_coefficients.jacobi(
            n, a, b, "monic"
        )
        assert all([a == 0 for a in alpha])
        assert (beta == [S(4) / 3, S(1) / 5, S(8) / 35, S(5) / 21, S(8) / 33]).all()
    else:
        a = 1.0
        b = 1.0
        tol = 1.0e-14
        _, _, alpha, beta = orthopy.line_segment.recurrence_coefficients.jacobi(
            n, a, b, "monic"
        )
        assert numpy.all(abs(alpha) < tol)
        assert numpy.all(
            abs(beta - [4.0 / 3.0, 1.0 / 5.0, 8.0 / 35.0, 5.0 / 21.0, 8.0 / 33.0]) < tol
        )
    return


if __name__ == "__main__":
    test_jacobi_monic(0, [1, 1, 1])
