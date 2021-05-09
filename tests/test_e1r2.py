import itertools

import pytest
import sympy
from sympy import Rational, pi, sqrt

import orthopy

x = sympy.Symbol("x")


# This function returns the integral of all monomials up to a given degree.
# See <https://github.com/nschloe/note-no-gamma>.
# def _integrate(f):
#     return sympy.integrate(f * sympy.exp(-(x ** 2 / 2)), (x, -oo, +oo)) / sqrt(2 * pi)
def _integrate_all_monomials_probabilists(max_k):
    out = []
    for k in range(max_k + 1):
        if k == 0:
            out.append(1)
        elif k == 1:
            out.append(0)
        else:
            out.append(out[k - 2] * (k - 1))
    return out


# def _integrate(f):
#     return sympy.integrate(f * sympy.exp(-(x ** 2)), (x, -oo, +oo))
def _integrate_all_monomials_physicists(max_k):
    out = []
    for k in range(max_k + 1):
        if k == 0:
            out.append(sqrt(pi))
        elif k == 1:
            out.append(0)
        else:
            out.append(out[k - 2] * Rational(k - 1, 2))
    return out


# Integrating polynomials is easily done by integrating the individual monomials and
# summing.
def _integrate_poly(p, standardization):
    coeffs = p.all_coeffs()[::-1]
    if standardization == "physicists":
        int_all_monomials = _integrate_all_monomials_physicists(p.degree())
    else:
        assert standardization == "probabilists"
        int_all_monomials = _integrate_all_monomials_probabilists(p.degree())
    return sum(coeff * mono_int for coeff, mono_int in zip(coeffs, int_all_monomials))


@pytest.mark.parametrize(
    "standardization,scaling,int0",
    [
        ("physicists", "classical", sqrt(pi)),
        ("physicists", "monic", sqrt(pi)),
        ("physicists", "normal", sqrt(sqrt(pi))),
        ("probabilists", "monic", 1),
        ("probabilists", "normal", 1),
    ],
)
def test_integral0(standardization, scaling, int0, n=4):
    p = sympy.poly(x)
    evaluator = orthopy.e1r2.Eval(p, standardization, scaling)

    assert _integrate_poly(next(evaluator), standardization) == int0
    for _ in range(n + 1):
        assert _integrate_poly(next(evaluator), standardization) == 0


@pytest.mark.parametrize("standardization", ["probabilists", "physicists"])
@pytest.mark.parametrize("scaling", ["classical", "monic", "normal"])
def test_orthogonality(standardization, scaling, n=4):
    p = sympy.poly(x)
    evaluator = orthopy.e1r2.Eval(p, standardization, scaling)
    vals = [next(evaluator) for _ in range(n + 1)]
    for f0, f1 in itertools.combinations(vals, 2):
        assert _integrate_poly(f0 * f1, standardization) == 0


@pytest.mark.parametrize("standardization", ["probabilists", "physicists"])
def test_normality(standardization, n=4):
    p = sympy.poly(x)
    iterator = orthopy.e1r2.Eval(p, standardization, "normal")
    for k, val in enumerate(itertools.islice(iterator, n)):
        if k == 0:
            val = sympy.poly(val, x)
        assert _integrate_poly(val ** 2, standardization) == 1


# @pytest.mark.parametrize("standardization", ["probabilists", "physicists"])
# def test_show(standardization):
#     orthopy.e1r2.show(4, standardization, "normal")


def test_show(n=5):
    orthopy.e1r2.show(n, "probabilists", "normal")
    orthopy.e1r2.savefig("e1r2.svg", n, "probabilists", "normal")


if __name__ == "__main__":
    test_show()
    # import matplotlib.pyplot as plt
    # plt.show()
    # plt.savefig("e1r.png", transparent=True)
