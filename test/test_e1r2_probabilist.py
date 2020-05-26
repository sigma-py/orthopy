import itertools

import numpy
import pytest
import sympy
from sympy import oo, pi, sqrt

import orthopy

standardization = "probabilist"
x = sympy.Symbol("x")


def _integrate(f):
    return sympy.integrate(f * sympy.exp(-(x ** 2 / 2)), (x, -oo, +oo)) / sqrt(2 * pi)


# This function returns the integral of all monomials up to a given degree.
# See <https://github.com/nschloe/note-no-gamma>.
def _integrate_all_monomials(max_k):
    out = []
    for k in range(max_k + 1):
        if k == 0:
            out.append(1)
        elif k == 1:
            out.append(0)
        else:
            out.append(out[k - 2] * (k - 1))
    return out


# Integrating polynomials is easily done by integrating the individual monomials and
# summing.
def _integrate_poly(p):
    coeffs = p.all_coeffs()[::-1]
    int_all_monomials = _integrate_all_monomials(p.degree())
    return sum(coeff * mono_int for coeff, mono_int in zip(coeffs, int_all_monomials))


@pytest.mark.parametrize("scaling,int0", [("monic", 1), ("normal", 1)])
def test_integral0(scaling, int0, n=4):
    p = sympy.poly(x)
    vals = orthopy.e1r2.tree(n, p, standardization, scaling, symbolic=True)
    vals[0] = sympy.poly(vals[0], x)

    assert _integrate_poly(vals[0]) == int0
    for val in vals[1:]:
        assert _integrate_poly(val) == 0


@pytest.mark.parametrize("scaling", ["monic", "normal"])
def test_orthogonality(scaling, n=4):
    p = sympy.poly(x)
    tree = orthopy.e1r2.tree(n, p, standardization, scaling, symbolic=True)
    vals = tree * numpy.roll(tree, 1, axis=0)
    for val in vals:
        assert _integrate_poly(val) == 0


def test_normality(n=4):
    p = sympy.poly(x)
    iterator = orthopy.e1r2.Iterator(p, standardization, "normal", symbolic=True)
    for k, val in enumerate(itertools.islice(iterator, n)):
        if k == 0:
            val = sympy.poly(val, x)
        assert _integrate_poly(val ** 2) == 1


def test_show():
    orthopy.e1r2.show(4, standardization, "normal")


if __name__ == "__main__":
    test_show()
