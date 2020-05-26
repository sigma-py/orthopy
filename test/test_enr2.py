import itertools

import numpy
import sympy
from sympy import oo

import orthopy

standaridization = "physicist"


def _integrate3(f, x, y, z):
    return sympy.integrate(
        f * sympy.exp(-(x ** 2 + y ** 2 + z ** 2)),
        (x, -oo, +oo),
        (y, -oo, +oo),
        (z, -oo, +oo),
    )


def test_integral0(n=3):
    X = sympy.symbols("x, y, z")
    vals = numpy.concatenate(orthopy.enr2.tree(n, X, standaridization, symbolic=True))

    assert _integrate3(vals[0], *X) == sympy.sqrt(sympy.sqrt(sympy.pi)) ** 3
    for val in vals[1:]:
        assert _integrate3(val, *X) == 0


def test_orthogonality(n=3):
    X = sympy.symbols("x, y, z")
    tree = numpy.concatenate(orthopy.enr2.tree(n, X, standaridization, symbolic=True))
    vals = tree * numpy.roll(tree, 1, axis=0)

    for val in vals:
        assert _integrate3(val, *X) == 0


def test_normality(n=3):
    X = sympy.symbols("x, y, z")
    iterator = orthopy.enr2.Iterator(X, standaridization, symbolic=True)
    for level in itertools.islice(iterator, n):
        for val in level:
            assert _integrate3(val ** 2, *X) == 1


if __name__ == "__main__":
    test_normality()
