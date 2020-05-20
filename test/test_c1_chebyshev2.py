import numpy
import pytest
import sympy
from sympy import Rational, S, pi, sqrt

import orthopy


@pytest.mark.parametrize(
    "n, y",
    [
        (0, [1, 1, 1]),
        (1, [0, Rational(1, 2), 1]),
        (2, [-Rational(1, 4), 0, Rational(3, 4)]),
        (3, [0, -Rational(1, 8), Rational(1, 2)]),
        (4, [Rational(1, 16), -Rational(1, 16), Rational(5, 16)]),
        (5, [0, 0, Rational(3, 16)]),
    ],
)
def test_chebyshev2_monic(n, y):
    x = numpy.array([0, Rational(1, 2), 1])

    out = orthopy.c1.recurrence_coefficients.chebyshev2(n, "monic", symbolic=True)

    # Test evaluation of one value
    y0 = orthopy.tools.line_evaluate(x[0], *out)
    assert y0 == y[0]

    # Test evaluation of multiple values
    val = orthopy.tools.line_evaluate(x, *out)
    assert all(val == y)


@pytest.mark.parametrize(
    "n, y",
    [
        (0, [1, 1, 1]),
        (1, [0, Rational(3, 4), Rational(3, 2)]),
        (2, [-Rational(5, 8), 0, Rational(15, 8)]),
        (3, [0, -Rational(35, 64), Rational(35, 16)]),
        (4, [Rational(63, 128), -Rational(63, 128), Rational(315, 128)]),
        (5, [0, 0, Rational(693, 256)]),
    ],
)
def test_chebyshev2_p11(n, y):
    x = numpy.array([0, Rational(1, 2), 1])

    out = orthopy.c1.recurrence_coefficients.chebyshev2(
        n, standardization="p(1)=(n+alpha over n)", symbolic=True
    )

    y0 = orthopy.tools.line_evaluate(x[0], *out)
    assert y0 == y[0]

    alpha = Rational(1, 2)
    assert sympy.binomial(n + alpha, n) == y[2]

    val = orthopy.tools.line_evaluate(x, *out)
    assert all(val == y)


@pytest.mark.parametrize(
    "n, y",
    [
        (0, [sqrt(2) / sqrt(pi), sqrt(2) / sqrt(pi), sqrt(2) / sqrt(pi)]),
        (1, [0, sqrt(2) / sqrt(pi), 2 * sqrt(2) / sqrt(pi)]),
        (2, [-sqrt(2) / sqrt(pi), 0, 3 * sqrt(2) / sqrt(pi)]),
        (3, [0, -sqrt(2) / sqrt(pi), 4 * sqrt(2) / sqrt(pi)]),
        (4, [sqrt(2) / sqrt(pi), -sqrt(2) / sqrt(pi), 5 * sqrt(2) / sqrt(pi)]),
        (5, [0, 0, 6 * sqrt(2) / sqrt(pi)]),
    ],
)
def test_chebyshev2_normal(n, y):
    x = numpy.array([0, S(1) / 2, 1])

    out = orthopy.c1.recurrence_coefficients.chebyshev2(
        n, standardization="normal", symbolic=True
    )

    y0 = orthopy.tools.line_evaluate(x[0], *out)
    assert y0 == y[0]

    val = orthopy.tools.line_evaluate(x, *out)
    assert all(val == y)


# sympy is too weak for the following tests
# def test_integral0(n=4):
#     x = sympy.Symbol("x")
#     vals = orthopy.c1.tree_chebyshev2(x, n, "normal", symbolic=True)
#
#     assert sympy.integrate(vals[0] * sqrt(1 - x ** 2), (x, -1, +1)) == sqrt(pi)/sqrt(2)
#     for val in vals[1:]:
#         assert sympy.integrate(val * sqrt(1 - x ** 2), (x, -1, +1)) == 0
#
#
# def test_normality(n=4):
#     x = sympy.Symbol("x")
#     vals = orthopy.c1.tree_chebyshev2(x, n, "normal", symbolic=True)
#
#     for val in vals:
#         assert sympy.integrate(val ** 2 * sqrt(1 - x ** 2), (x, -1, +1)) == 1
#
#
# def test_orthogonality(n=4):
#     x = sympy.Symbol("x")
#     vals = orthopy.c1.tree_chebyshev2(x, n, "normal", symbolic=True)
#     out = vals * numpy.roll(vals, 1, axis=0)
#
#     for val in out:
#         assert sympy.simplify(sympy.integrate(val * sqrt(1 - x ** 2), (x, -1, +1))) == 0


@pytest.mark.parametrize("t, ref", [(Rational(1, 2), 0), (1, Rational(3, 16))])
def test_eval(t, ref, tol=1.0e-14):
    n = 5
    p0, a, b, c = orthopy.c1.recurrence_coefficients.chebyshev2(
        n, "monic", symbolic=True
    )
    value = orthopy.tools.line_evaluate(t, p0, a, b, c)

    assert value == ref


@pytest.mark.parametrize(
    "t, ref",
    [
        (numpy.array([1]), numpy.array([Rational(3, 16)])),
        (numpy.array([1, 2]), numpy.array([Rational(3, 16), Rational(195, 8)])),
    ],
)
def test_eval_vec(t, ref, tol=1.0e-14):
    n = 5
    p0, a, b, c = orthopy.c1.recurrence_coefficients.chebyshev2(
        n, "monic", symbolic=True
    )
    value = orthopy.tools.line_evaluate(t, p0, a, b, c)

    assert (value == ref).all()


def test_plot(n=4):
    orthopy.c1.plot(n, +0.5, +0.5)


if __name__ == "__main__":
    test_plot()
    import matplotlib.pyplot as plt

    # plt.show()
    plt.savefig("line-segment-chebyshev2.png", transparent=True)
