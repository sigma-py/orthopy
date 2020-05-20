import numpy
import pytest
from sympy import S, sqrt

import orthopy
from helpers import get_nth


@pytest.mark.parametrize(
    "standardization, n, y",
    [
        ("monic", 0, [1, 1, 1]),
        ("monic", 1, [S(1) / 7, S(9) / 14, S(8) / 7]),
        ("monic", 2, [-S(1) / 9, S(1) / 4, S(10) / 9]),
        ("monic", 3, [-S(1) / 33, S(7) / 264, S(32) / 33]),
        ("monic", 4, [S(3) / 143, -S(81) / 2288, S(112) / 143]),
        ("monic", 5, [S(1) / 143, -S(111) / 4576, S(256) / 429]),
        #
        ("p(1)=(n+alpha over n)", 0, [1, 1, 1]),
        ("p(1)=(n+alpha over n)", 1, [S(1) / 2, S(9) / 4, 4]),
        ("p(1)=(n+alpha over n)", 2, [-1, S(9) / 4, 10]),
        ("p(1)=(n+alpha over n)", 3, [-S(5) / 8, S(35) / 64, 20]),
        ("p(1)=(n+alpha over n)", 4, [S(15) / 16, -S(405) / 256, 35]),
        ("p(1)=(n+alpha over n)", 5, [S(21) / 32, -S(2331) / 1024, 56]),
        #
        ("normal", 0, [sqrt(15) / 4, sqrt(15) / 4, sqrt(15) / 4]),
        ("normal", 1, [sqrt(10) / 8, 9 * sqrt(10) / 16, sqrt(10)]),
        ("normal", 2, [-sqrt(35) / 8, 9 * sqrt(35) / 32, 5 * sqrt(35) / 4]),
        ("normal", 3, [-sqrt(210) / 32, 7 * sqrt(210) / 256, sqrt(210)]),
        ("normal", 4, [3 * sqrt(210) / 64, -81 * sqrt(210) / 1024, 7 * sqrt(210) / 4]),
        ("normal", 5, [3 * sqrt(105) / 64, -333 * sqrt(105) / 2048, 4 * sqrt(105)]),
    ],
)
def test_jacobi_monic(standardization, n, y):
    x = numpy.array([0, S(1) / 2, 1])

    alpha = 3
    beta = 2
    symbolic = True

    y2 = get_nth(
        orthopy.c1.jacobi.Iterator(x[2], alpha, beta, standardization, symbolic), n
    )
    assert y2 == y[2]

    val = get_nth(
        orthopy.c1.jacobi.Iterator(x, alpha, beta, standardization, symbolic), n
    )
    assert all(val == y)


@pytest.mark.parametrize("dtype", [numpy.float, S])
def test_jacobi(dtype):
    n = 5
    if dtype == S:
        a = S(1)
        b = S(1)
        _, _, alpha, beta = orthopy.c1.recurrence_coefficients.jacobi(
            n, a, b, "monic", symbolic=True
        )
        assert all([a == 0 for a in alpha])
        assert (beta == [S(4) / 3, S(1) / 5, S(8) / 35, S(5) / 21, S(8) / 33]).all()
    else:
        a = 1.0
        b = 1.0
        tol = 1.0e-14
        _, _, alpha, beta = orthopy.c1.recurrence_coefficients.jacobi(
            n, a, b, "monic", symbolic=False
        )
        assert numpy.all(abs(alpha) < tol)
        assert numpy.all(
            abs(beta - [4.0 / 3.0, 1.0 / 5.0, 8.0 / 35.0, 5.0 / 21.0, 8.0 / 33.0]) < tol
        )


if __name__ == "__main__":
    test_jacobi_monic(0, [1, 1, 1])
