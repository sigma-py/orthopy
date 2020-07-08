import itertools

import numpy
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
        (2, [-Rational(1, 2), -Rational(1, 4), Rational(1, 2)]),
        (3, [0, -Rational(1, 4), Rational(1, 4)]),
        (4, [Rational(1, 8), -Rational(1, 16), Rational(1, 8)]),
        (5, [0, Rational(1, 32), Rational(1, 16)]),
    ],
)
def test_chebyshev1_monic(n, y):
    x = numpy.array([0, Rational(1, 2), 1])

    # Test evaluation of one value
    y0 = get_nth(orthopy.c1.chebyshev1.Eval(x[0], "monic", symbolic=True), n)
    assert y0 == y[0]

    # Test evaluation of multiple values
    val = get_nth(orthopy.c1.chebyshev1.Eval(x, "monic", symbolic=True), n)
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

    y0 = get_nth(orthopy.c1.chebyshev1.Eval(x[0], scaling), n)
    assert y0 == y[0]

    alpha = -Rational(1, 2)
    assert sympy.binomial(n + alpha, n) == y[2]

    val = get_nth(orthopy.c1.chebyshev1.Eval(x, scaling), n)
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

    y0 = get_nth(orthopy.c1.chebyshev1.Eval(x[0], scaling, symbolic=True), n)
    assert y0 == y[0]

    val = get_nth(orthopy.c1.chebyshev1.Eval(x, scaling, symbolic=True), n)
    assert all(val == y)


# def _integrate(f, x):
#     # expanding makes sympy work a lot faster here
#     return sympy.integrate(sympy.expand(f) / sqrt(1 - x ** 2), (x, -1, +1))


# This function returns the integral of all monomials up to a given degree.
# See <https://github.com/nschloe/note-no-gamma>.
def _integrate_all_monomials(max_k):
    out = []
    for k in range(max_k + 1):
        if k == 0:
            out.append(pi)
        elif k == 1:
            out.append(0)
        else:
            out.append(out[k - 2] * sympy.Rational(k - 1, k))
    return out


# Integrating polynomials is easily done by integrating the individual monomials and
# summing.
def _integrate_poly(p):
    coeffs = p.all_coeffs()[::-1]
    int_all_monomials = _integrate_all_monomials(p.degree())
    return sum(coeff * mono_int for coeff, mono_int in zip(coeffs, int_all_monomials))


def test_integral0(n=4):
    x = sympy.Symbol("x")
    p = sympy.poly(x)
    evaluator = orthopy.c1.chebyshev1.Eval(p, "normal")

    assert _integrate_poly(next(evaluator)) == sqrt(pi)
    for _ in range(n + 1):
        assert _integrate_poly(next(evaluator)) == 0


def test_normality(n=4):
    x = sympy.Symbol("x")
    p = sympy.poly(x, x)
    iterator = orthopy.c1.chebyshev1.Eval(p, "normal")
    for k, val in enumerate(itertools.islice(iterator, 5)):
        if k == 0:
            val = sympy.poly(val, x)
        assert _integrate_poly(val ** 2) == 1


def test_orthogonality(n=4):
    x = sympy.Symbol("x")
    p = sympy.poly(x)
    evaluator = orthopy.c1.chebyshev1.Eval(p, "normal")
    vals = [next(evaluator) for _ in range(n + 1)]
    for f0, f1 in itertools.combinations(vals, 2):
        assert _integrate_poly(f0 * f1) == 0


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
    value = get_nth(orthopy.c1.chebyshev1.Eval(t, "monic"), n)
    assert numpy.all(value == ref)


def test_show(n=5):
    orthopy.c1.chebyshev1.show(n, "normal")
    orthopy.c1.chebyshev1.savefig("chebyshev1.svg", n, "normal")


if __name__ == "__main__":
    test_show()
