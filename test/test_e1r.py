import numpy
import pytest
import sympy
from sympy import oo

import orthopy


@pytest.mark.parametrize("alpha", [0, 1])
def test_integral0(alpha, n=4):
    x = sympy.Symbol("x")
    vals = numpy.concatenate(
        orthopy.e1r.tree(numpy.array([x]), n, alpha=alpha, symbolic=True)
    )

    assert sympy.integrate(vals[0] * x ** alpha * sympy.exp(-x), (x, 0, +oo)) == 1

    for val in vals[1:]:
        assert sympy.integrate(val * x ** alpha * sympy.exp(-x), (x, 0, +oo)) == 0


@pytest.mark.parametrize("alpha", [0, 1])
@pytest.mark.parametrize("standardization", ["monic", "classical", "normal"])
def test_orthogonality(alpha, standardization, n=4):
    x = sympy.Symbol("x")
    tree = numpy.concatenate(
        orthopy.e1r.tree(
            numpy.array([x]),
            n,
            alpha=alpha,
            standardization=standardization,
            symbolic=True,
        )
    )
    vals = tree * numpy.roll(tree, 1, axis=0)

    for val in vals:
        assert sympy.integrate(val * x ** alpha * sympy.exp(-x), (x, 0, +oo)) == 0


@pytest.mark.parametrize("alpha", [0, 1])
def test_normality(alpha, n=4):
    x = sympy.Symbol("x")
    tree = numpy.concatenate(
        orthopy.e1r.tree(numpy.array([x]), n, alpha=alpha, symbolic=True)
    )

    for val in tree:
        assert sympy.integrate(val ** 2 * x ** alpha * sympy.exp(-x), (x, 0, +oo)) == 1


def test_show():
    orthopy.e1r.show(L=4)


if __name__ == "__main__":
    test_show()
    # import matplotlib.pyplot as plt
    # plt.show()
    # plt.savefig("e1r.png", transparent=True)
