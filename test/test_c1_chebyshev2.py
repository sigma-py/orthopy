import itertools

import numpy as np
import pytest
import sympy
from helpers import get_nth
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
    x = np.array([0, Rational(1, 2), 1])

    scaling = "monic"

    # Test evaluation of one value
    y0 = get_nth(orthopy.c1.chebyshev2.Eval(x[0], scaling, symbolic=True), n)
    assert y0 == y[0]

    # Test evaluation of multiple values
    val = get_nth(orthopy.c1.chebyshev2.Eval(x, scaling, symbolic=True), n)
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
    x = np.array([0, Rational(1, 2), 1])

    scaling = "classical"

    y0 = get_nth(orthopy.c1.chebyshev2.Eval(x[0], scaling, symbolic=True), n)
    assert y0 == y[0]

    alpha = Rational(1, 2)
    assert sympy.binomial(n + alpha, n) == y[2]

    val = get_nth(orthopy.c1.chebyshev2.Eval(x, scaling, symbolic=True), n)
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
    x = np.array([0, S(1) / 2, 1])

    scaling = "normal"

    y0 = get_nth(orthopy.c1.chebyshev2.Eval(x[0], scaling, symbolic=True), n)
    assert y0 == y[0]

    val = get_nth(orthopy.c1.chebyshev2.Eval(x, scaling, symbolic=True), n)
    assert all(val == y)


# def _integrate(f, x):
#     return sympy.integrate(f * sqrt(1 - x ** 2), (x, -1, +1))


# This function returns the integral of all monomials up to a given degree.
# See <https://github.com/nschloe/note-no-gamma>.
def _integrate_all_monomials(max_k):
    out = []
    for k in range(max_k + 1):
        if k == 0:
            out.append(pi / 2)
        elif k == 1:
            out.append(0)
        else:
            out.append(out[k - 2] * sympy.Rational(k - 1, k + 2))
    return out


# Integrating polynomials is easily done by integrating the individual monomials and
# summing.
def _integrate_poly(p):
    coeffs = p.all_coeffs()[::-1]
    int_all_monomials = _integrate_all_monomials(p.degree())
    return sum(coeff * mono_int for coeff, mono_int in zip(coeffs, int_all_monomials))


# sympy is too weak for the following tests
def test_integral0(n=4):
    x = sympy.Symbol("x")
    p = sympy.poly(x)
    evaluator = orthopy.c1.chebyshev2.Eval(p, "normal")

    assert _integrate_poly(next(evaluator)) == sqrt(pi) / sqrt(2)
    for _ in range(n + 1):
        assert _integrate_poly(next(evaluator)) == 0


def test_normality(n=4):
    x = sympy.Symbol("x")
    p = sympy.poly(x)

    iterator = orthopy.c1.chebyshev2.Eval(p, "normal")
    for k, val in enumerate(itertools.islice(iterator, n)):
        if k == 0:
            val = sympy.poly(val, x)
        # sympy integration isn't only slow, but also wrong,
        # <https://github.com/sympy/sympy/issues/19427>
        # assert _integrate(val ** 2, x) == 1
        assert _integrate_poly(val ** 2) == 1


def test_orthogonality(n=4):
    x = sympy.Symbol("x")
    p = sympy.poly(x)
    evaluator = orthopy.c1.chebyshev2.Eval(p, "normal")
    vals = [next(evaluator) for _ in range(n + 1)]
    for f0, f1 in itertools.combinations(vals, 2):
        assert _integrate_poly(f0 * f1) == 0


@pytest.mark.parametrize(
    "t, ref",
    [
        (Rational(1, 2), 0),
        (1, Rational(3, 16)),
        (np.array([1]), np.array([Rational(3, 16)])),
        (np.array([1, 2]), np.array([Rational(3, 16), Rational(195, 8)])),
    ],
)
def test_eval(t, ref):
    n = 5
    value = get_nth(orthopy.c1.chebyshev2.Eval(t, "monic", symbolic=True), n)
    assert np.all(value == ref)


def test_show(n=5):
    orthopy.c1.chebyshev2.show(n, "normal")
    orthopy.c1.chebyshev2.savefig("chebyshev2.svg", n, "normal")


if __name__ == "__main__":
    test_show()
