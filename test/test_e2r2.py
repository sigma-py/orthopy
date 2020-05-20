import numpy
import sympy
from sympy import oo

import orthopy


def _integrate(f, x, y):
    return sympy.integrate(
        f * sympy.exp(-(x ** 2) - y ** 2), (x, -oo, +oo), (y, -oo, +oo)
    )


def test_integral0(n=4):
    """Make sure that the polynomials are orthonormal
    """
    x = sympy.Symbol("x")
    y = sympy.Symbol("y")
    vals = numpy.concatenate(orthopy.e2r2.tree(numpy.array([x, y]), n, symbolic=True))

    assert _integrate(vals[0], x, y) == sympy.sqrt(sympy.pi)
    for val in vals[1:]:
        assert _integrate(val, x, y) == 0


def test_orthogonality(n=4):
    x = sympy.Symbol("x")
    y = sympy.Symbol("y")
    tree = numpy.concatenate(orthopy.e2r2.tree(numpy.array([x, y]), n, symbolic=True))
    vals = tree * numpy.roll(tree, 1, axis=0)

    for val in vals:
        assert _integrate(val, x, y) == 0


def test_normality(n=4):
    x = sympy.Symbol("x")
    y = sympy.Symbol("y")
    tree = numpy.concatenate(orthopy.e2r2.tree(numpy.array([x, y]), n, symbolic=True))

    for val in tree:
        assert _integrate(val ** 2, x, y) == 1


def test_show(n=2, r=1, d=1.5):
    def f(X):
        return orthopy.e2r2.tree(X, n)[n][r]

    orthopy.e2r2.show(f, d=d)
    # orthopy.e2r2.plot(f, d=d)
    # import matplotlib.pyplot as plt
    # plt.savefig('e2r2.png', transparent=True)


if __name__ == "__main__":
    test_show(n=1, r=0)
