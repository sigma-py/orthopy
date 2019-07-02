import numpy
import pytest
import sympy
from sympy import oo

import orthopy


def test_integral0(n=4):
    x = sympy.Symbol("x")
    vals = numpy.concatenate(
        orthopy.e1r2.tree(numpy.array([x]), n, "normal", symbolic=True)
    )

    assert sympy.integrate(vals[0] * sympy.exp(-x ** 2), (x, -oo, +oo)) == sympy.sqrt(
        sympy.sqrt(sympy.pi)
    )

    for val in vals[1:]:
        assert sympy.integrate(val * sympy.exp(-x ** 2), (x, -oo, +oo)) == 0
    return


@pytest.mark.parametrize("standardization", ["monic", "physicist", "normal"])
def test_orthogonality(standardization, n=4):
    x = sympy.Symbol("x")
    tree = orthopy.e1r2.tree(numpy.array(x), n, standardization, symbolic=True)
    vals = tree * numpy.roll(tree, 1, axis=0)

    if standardization == "monic":
        weight_function = sympy.exp(-x ** 2 / 2)
    else:
        weight_function = sympy.exp(-x ** 2)

    for val in vals:
        assert sympy.integrate(val * weight_function, (x, -oo, +oo)) == 0
    return


def test_normality(n=4):
    x = sympy.Symbol("x")
    tree = numpy.concatenate(
        orthopy.e1r2.tree(numpy.array([x]), n, "normal", symbolic=True)
    )

    for val in tree:
        assert sympy.integrate(val ** 2 * sympy.exp(-x ** 2), (x, -oo, +oo)) == 1
    return


def test_show():
    orthopy.e1r2.show(L=4)
    return


if __name__ == "__main__":
    test_show()
