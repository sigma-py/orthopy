import numpy
import sympy

import orthopy


def test_integral0(n=4):
    """Make sure that the polynomials are orthonormal
    """
    x = sympy.Symbol("x")
    y = sympy.Symbol("y")
    z = sympy.Symbol("z")
    vals = numpy.concatenate(orthopy.c3.tree(numpy.array([x, y, z]), n, symbolic=True))

    assert (
        sympy.integrate(vals[0], (x, -1, +1), (y, -1, +1), (z, -1, +1))
        == sympy.sqrt(2) ** 3
    )
    for val in vals[1:]:
        assert sympy.integrate(val, (x, -1, +1), (y, -1, +1), (z, -1, +1)) == 0


def test_orthogonality(n=4):
    x = sympy.Symbol("x")
    y = sympy.Symbol("y")
    z = sympy.Symbol("z")
    tree = numpy.concatenate(orthopy.c3.tree(numpy.array([x, y, z]), n, symbolic=True))
    vals = tree * numpy.roll(tree, 1, axis=0)

    for val in vals:
        assert sympy.integrate(val, (x, -1, +1), (y, -1, +1), (z, -1, +1)) == 0


def test_normality(n=4):
    x = sympy.Symbol("x")
    y = sympy.Symbol("y")
    z = sympy.Symbol("z")
    tree = numpy.concatenate(orthopy.c3.tree(numpy.array([x, y, z]), n, symbolic=True))

    for val in tree:
        assert sympy.integrate(val ** 2, (x, -1, +1), (y, -1, +1), (z, -1, +1)) == 1


def test_write():
    orthopy.c3.write("hexa.vtk", lambda X: orthopy.c3.tree(X.T, 5)[5][5])


if __name__ == "__main__":
    test_write()
