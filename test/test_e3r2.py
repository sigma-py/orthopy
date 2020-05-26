import numpy
import sympy
from sympy import oo

import orthopy

X = sympy.symbols("x, y, z")

standardization = "physicist"


def _integrate(f):
    ranges = [(x, -oo, +oo) for x in X]
    return sympy.integrate(f * sympy.exp(-sum(x ** 2 for x in X)), *ranges)


def test_integral0(n=3):
    """Make sure that the polynomials are orthonormal
    """
    vals = numpy.concatenate(orthopy.e3r2.tree(n, X, standardization, symbolic=True))

    assert _integrate(vals[0]) == sympy.sqrt(sympy.sqrt(sympy.pi)) ** 3
    for val in vals[1:]:
        assert _integrate(val) == 0


def test_orthogonality(n=3):
    tree = numpy.concatenate(orthopy.e3r2.tree(n, X, standardization, symbolic=True))
    vals = tree * numpy.roll(tree, 1, axis=0)

    for val in vals:
        assert _integrate(val) == 0


def test_normality(n=3):
    tree = numpy.concatenate(orthopy.e3r2.tree(n, X, standardization, symbolic=True))

    for val in tree:
        assert _integrate(val ** 2) == 1


def test_write(n=5, r=5):
    orthopy.e3r2.write(
        "e3r2.vtk", lambda X: orthopy.e3r2.tree(n, X.T, standardization)[n][r]
    )


if __name__ == "__main__":
    test_write(n=1, r=0)
