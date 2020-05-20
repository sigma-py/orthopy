import numpy
import sympy

import orthopy


def _integrate(f, x, y, z):
    return sympy.integrate(f, (x, -1, +1), (y, -1, +1), (z, -1, +1))


def test_integral0(n=4):
    """Make sure that the polynomials are orthonormal
    """
    xyz = sympy.symbols("x, y, z")
    vals = numpy.concatenate(orthopy.c3.tree(n, xyz, symbolic=True))

    assert _integrate(vals[0], *xyz) == sympy.sqrt(2) ** 3
    for val in vals[1:]:
        assert _integrate(val, *xyz) == 0


def test_orthogonality(n=4):
    xyz = sympy.symbols("x, y, z")
    tree = numpy.concatenate(orthopy.c3.tree(n, xyz, symbolic=True))
    vals = tree * numpy.roll(tree, 1, axis=0)
    for val in vals:
        assert _integrate(val, *xyz) == 0


def test_normality(n=4):
    xyz = sympy.symbols("x, y, z")
    tree = numpy.concatenate(orthopy.c3.tree(n, xyz, symbolic=True))
    for val in tree:
        assert _integrate(val ** 2, *xyz) == 1


def test_write():
    orthopy.c3.write("hexa.vtk", lambda X: orthopy.c3.tree(5, X.T)[5][5])


if __name__ == "__main__":
    test_write()
