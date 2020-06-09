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
def _integrate_all_monomials_probabilist(max_k):
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
def _integrate_all_monomials_physicist(max_k):
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
    if standardization == "physicist":
        int_all_monomials = _integrate_all_monomials_physicist(p.degree())
    else:
        assert standardization == "probabilist"
        int_all_monomials = _integrate_all_monomials_probabilist(p.degree())
    return sum(coeff * mono_int for coeff, mono_int in zip(coeffs, int_all_monomials))


@pytest.mark.parametrize(
    "standardization,scaling,int0",
    [
        ("physicist", "classical", sqrt(pi)),
        ("physicist", "monic", sqrt(pi)),
        ("physicist", "normal", sqrt(sqrt(pi))),
        ("probabilist", "monic", 1),
        ("probabilist", "normal", 1),
    ],
)
def test_integral0(standardization, scaling, int0, n=4):
    p = sympy.poly(x)
    vals = orthopy.e1r2.tree(n, p, standardization, scaling, symbolic=True)

    assert _integrate_poly(vals[0], standardization) == int0
    for val in vals[1:]:
        assert _integrate_poly(val, standardization) == 0


@pytest.mark.parametrize("standardization", ["probabilist", "physicist"])
@pytest.mark.parametrize("scaling", ["classical", "monic", "normal"])
def test_orthogonality(standardization, scaling, n=4):
    p = sympy.poly(x)
    tree = orthopy.e1r2.tree(n, p, standardization, scaling, symbolic=True)
    for f0, f1 in itertools.combinations(tree, 2):
        assert _integrate_poly(f0 * f1, standardization) == 0


@pytest.mark.parametrize("standardization", ["probabilist", "physicist"])
def test_normality(standardization, n=4):
    p = sympy.poly(x)
    iterator = orthopy.e1r2.Eval(p, standardization, "normal", symbolic=True)
    for k, val in enumerate(itertools.islice(iterator, n)):
        if k == 0:
            val = sympy.poly(val, x)
        assert _integrate_poly(val ** 2, standardization) == 1


@pytest.mark.parametrize("standardization", ["probabilist", "physicist"])
def test_show(standardization):
    orthopy.e1r2.show(4, standardization, "normal")


if __name__ == "__main__":
    test_show()
