import numpy
import pytest
import sympy
from sympy import oo

import orthopy


def _integrate(f, x):
    return sympy.integrate(f * sympy.exp(-(x ** 2)), (x, -oo, +oo))


def test_integral0(n=4):
    x = sympy.Symbol("x")
    vals = orthopy.e1r2.tree(n, x, "normal", symbolic=True)

    assert _integrate(vals[0], x) == sympy.sqrt(sympy.sqrt(sympy.pi))
    for val in vals[1:]:
        assert _integrate(val, x) == 0


@pytest.mark.parametrize("scaling", ["monic", "physicist", "normal"])
def test_orthogonality(scaling, n=4):
    x = sympy.Symbol("x")
    tree = orthopy.e1r2.tree(n, x, scaling, symbolic=True)
    vals = tree * numpy.roll(tree, 1, axis=0)

    if scaling == "monic":
        weight_function = sympy.exp(-(x ** 2) / 2)
    else:
        weight_function = sympy.exp(-(x ** 2))

    for val in vals:
        assert sympy.integrate(val * weight_function, (x, -oo, +oo)) == 0


def test_normality(n=4):
    x = sympy.Symbol("x")
    tree = orthopy.e1r2.tree(n, x, "normal", symbolic=True)
    for val in tree:
        assert _integrate(val ** 2, x) == 1


def test_show():
    orthopy.e1r2.show(L=4)


if __name__ == "__main__":
    test_show()
