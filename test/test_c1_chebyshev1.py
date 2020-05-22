import numpy
import pytest
import sympy
from sympy import Rational, S, pi, sqrt

import orthopy
from helpers import get_nth


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
def test_chebyshev1_monic(n, y):
    x = numpy.array([0, Rational(1, 2), 1])

    # Test evaluation of one value
    y0 = get_nth(orthopy.c1.chebyshev1.Iterator(x[0], "monic", symbolic=True), n)
    assert y0 == y[0]

    # Test evaluation of multiple values
    val = get_nth(orthopy.c1.chebyshev1.Iterator(x, "monic", symbolic=True), n)
    assert all(val == y)


@pytest.mark.parametrize(
    "n, y",
    [
        (0, [1, 1, 1]),
        (1, [0, Rational(1, 4), Rational(1, 2)]),
        (2, [-Rational(3, 8), -Rational(3, 16), Rational(3, 8)]),
        (3, [0, -Rational(5, 16), Rational(5, 16)]),
        (4, [Rational(35, 128), -Rational(35, 256), Rational(35, 128)]),
        (5, [0, Rational(63, 512), Rational(63, 256)]),
    ],
)
def test_chebyshev1_p11(n, y):
    x = numpy.array([0, Rational(1, 2), 1])

    scaling = "classical"

    y0 = get_nth(orthopy.c1.chebyshev1.Iterator(x[0], scaling, symbolic=True), n)
    assert y0 == y[0]

    alpha = -Rational(1, 2)
    assert sympy.binomial(n + alpha, n) == y[2]

    val = get_nth(orthopy.c1.chebyshev1.Iterator(x, scaling, symbolic=True), n)
    assert all(val == y)


@pytest.mark.parametrize(
    "n, y",
    [
        (0, [1 / sqrt(pi), 1 / sqrt(pi), 1 / sqrt(pi)]),
        (1, [0, sqrt(2) / (2 * sqrt(pi)), sqrt(2) / sqrt(pi)]),
        (2, [-sqrt(2) / sqrt(pi), -sqrt(2) / (2 * sqrt(pi)), sqrt(2) / sqrt(pi)]),
        (3, [0, -sqrt(2) / sqrt(pi), sqrt(2) / sqrt(pi)]),
        (4, [sqrt(2) / sqrt(pi), -sqrt(2) / (2 * sqrt(pi)), sqrt(2) / sqrt(pi)]),
        (5, [0, sqrt(2) / (2 * sqrt(pi)), sqrt(2) / sqrt(pi)]),
    ],
)
def test_chebyshev1_normal(n, y):
    x = numpy.array([0, S(1) / 2, 1])

    scaling = "normal"

    y0 = get_nth(orthopy.c1.chebyshev1.Iterator(x[0], scaling, symbolic=True), n)
    assert y0 == y[0]

    val = get_nth(orthopy.c1.chebyshev1.Iterator(x, scaling, symbolic=True), n)
    assert all(val == y)


def _integrate(f, x):
    return sympy.integrate(f / sqrt(1 - x ** 2), (x, -1, +1))


def test_integral0(n=4):
    x = sympy.Symbol("x")
    vals = orthopy.c1.chebyshev1.tree(n, x, "normal", symbolic=True)

    assert _integrate(vals[0], x) == sqrt(pi)
    for val in vals[1:]:
        assert _integrate(val, x) == 0


def test_normality(n=4):
    x = sympy.Symbol("x")
    for k, val in enumerate(orthopy.c1.chebyshev1.Iterator(x, "normal", symbolic=True)):
        assert _integrate(val ** 2, x) == 1
        if k == n:
            break


def test_orthogonality(n=4):
    x = sympy.Symbol("x")
    vals = orthopy.c1.chebyshev1.tree(n, x, "normal", symbolic=True)
    out = vals * numpy.roll(vals, 1, axis=0)

    for val in out:
        assert _integrate(val, x) == 0


@pytest.mark.parametrize(
    "t, ref",
    [
        (Rational(1, 2), Rational(1, 32)),
        (1, Rational(1, 16)),
        (numpy.array([1]), numpy.array([Rational(1, 16)])),
        (numpy.array([1, 2]), numpy.array([Rational(1, 16), Rational(181, 8)])),
    ],
)
def test_eval(t, ref, tol=1.0e-14):
    n = 5
    value = get_nth(orthopy.c1.chebyshev1.Iterator(t, "monic", symbolic=True), n)
    assert numpy.all(value == ref)


def test_plot(n=4):
    orthopy.c1.plot(n, -0.5, -0.5)


if __name__ == "__main__":
    test_plot()
    import matplotlib.pyplot as plt

    # plt.show()
    plt.savefig("line-segment-chebyshev1.png", transparent=True)
