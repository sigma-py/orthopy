import numpy
import pytest
import sympy
from sympy import oo, pi, sqrt

import orthopy


standardization = "probabilist"


def _integrate(f, x):
    return sympy.integrate(f * sympy.exp(-(x ** 2 / 2)), (x, -oo, +oo)) / sqrt(2 * pi)


def test_integral0(n=4):
    x = sympy.Symbol("x")
    vals = orthopy.e1r2.tree(n, x, standardization, "normal", symbolic=True)

    assert _integrate(vals[0], x) == 1
    for val in vals[1:]:
        assert _integrate(val, x) == 0


@pytest.mark.parametrize("scaling", ["monic", "normal"])
def test_orthogonality(scaling, n=4):
    x = sympy.Symbol("x")
    tree = orthopy.e1r2.tree(n, x, standardization, scaling, symbolic=True)
    vals = tree * numpy.roll(tree, 1, axis=0)

    for val in vals:
        assert _integrate(val, x) == 0


def test_normality(n=4):
    x = sympy.Symbol("x")
    tree = orthopy.e1r2.tree(n, x, standardization, "normal", symbolic=True)
    for val in tree:
        assert _integrate(val ** 2, x) == 1


def test_show():
    orthopy.e1r2.show(L=4)


if __name__ == "__main__":
    test_show()
