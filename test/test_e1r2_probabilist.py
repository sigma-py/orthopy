import numpy
import pytest
import sympy
from sympy import oo, pi, sqrt

import orthopy


standardization = "probabilist"
x = sympy.Symbol("x")


def _integrate(f):
    return sympy.integrate(f * sympy.exp(-(x ** 2 / 2)), (x, -oo, +oo)) / sqrt(2 * pi)


@pytest.mark.parametrize("scaling,int0", [
    ("monic", 1),
    ("normal", 1),
])
def test_integral0(scaling, int0, n=4):
    vals = orthopy.e1r2.tree(n, x, standardization, scaling, symbolic=True)
    assert _integrate(vals[0]) == int0
    for val in vals[1:]:
        assert _integrate(val) == 0


@pytest.mark.parametrize("scaling", ["monic", "normal"])
def test_orthogonality(scaling, n=4):
    tree = orthopy.e1r2.tree(n, x, standardization, scaling, symbolic=True)
    vals = tree * numpy.roll(tree, 1, axis=0)
    for val in vals:
        assert _integrate(val) == 0


def test_normality(n=4):
    tree = orthopy.e1r2.tree(n, x, standardization, "normal", symbolic=True)
    for val in tree:
        assert _integrate(val ** 2) == 1


def test_show():
    orthopy.e1r2.show(L=4)


if __name__ == "__main__":
    test_show()
