import numpy
import sympy
from sympy import oo

import orthopy


def test_integral0(n=4):
    """Make sure that the polynomials are orthonormal
    """
    x = sympy.Symbol("x")
    y = sympy.Symbol("y")
    z = sympy.Symbol("z")
    vals = numpy.concatenate(
        orthopy.enr2.tree(numpy.array([x, y, z]), n, symbolic=True)
    )

    assert (
        sympy.integrate(
            vals[0] * sympy.exp(-(x ** 2) - y ** 2 - z ** 2),
            (x, -oo, +oo),
            (y, -oo, +oo),
            (z, -oo, +oo),
        )
        == sympy.sqrt(sympy.sqrt(sympy.pi)) ** 3
    )
    for val in vals[1:]:
        assert (
            sympy.integrate(
                val * sympy.exp(-(x ** 2) - y ** 2 - z ** 2),
                (x, -oo, +oo),
                (y, -oo, +oo),
                (z, -oo, +oo),
            )
            == 0
        )
    return


def test_orthogonality(n=4):
    x = sympy.Symbol("x")
    y = sympy.Symbol("y")
    z = sympy.Symbol("z")
    tree = numpy.concatenate(
        orthopy.enr2.tree(numpy.array([x, y, z]), n, symbolic=True)
    )
    vals = tree * numpy.roll(tree, 1, axis=0)

    for val in vals:
        assert (
            sympy.integrate(
                val * sympy.exp(-(x ** 2) - y ** 2 - z ** 2),
                (x, -oo, +oo),
                (y, -oo, +oo),
                (z, -oo, +oo),
            )
            == 0
        )
    return


def test_normality(n=4):
    x = sympy.Symbol("x")
    y = sympy.Symbol("y")
    z = sympy.Symbol("z")
    tree = numpy.concatenate(
        orthopy.enr2.tree(numpy.array([x, y, z]), n, symbolic=True)
    )

    for val in tree:
        assert (
            sympy.integrate(
                val ** 2 * sympy.exp(-(x ** 2) - y ** 2 - z ** 2),
                (x, -oo, +oo),
                (y, -oo, +oo),
                (z, -oo, +oo),
            )
            == 1
        )
    return


if __name__ == "__main__":
    test_normality()
