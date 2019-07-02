import numpy
import sympy

import orthopy


def test_integral0(n=4):
    """Make sure that the polynomials are orthonormal
    """
    x = sympy.Symbol("x")
    y = sympy.Symbol("y")
    vals = numpy.concatenate(
        orthopy.quadrilateral.tree(numpy.array([x, y]), n, symbolic=True)
    )

    assert sympy.integrate(vals[0], (x, -1, +1), (y, -1, +1)) == 2
    for val in vals[1:]:
        assert sympy.integrate(val, (x, -1, +1), (y, -1, +1)) == 0
    return


def test_orthogonality(n=4):
    x = sympy.Symbol("x")
    y = sympy.Symbol("y")
    tree = numpy.concatenate(
        orthopy.quadrilateral.tree(numpy.array([x, y]), n, symbolic=True)
    )
    vals = tree * numpy.roll(tree, 1, axis=0)

    for val in vals:
        assert sympy.integrate(val, (x, -1, +1), (y, -1, +1)) == 0
    return


def test_normality(n=4):
    x = sympy.Symbol("x")
    y = sympy.Symbol("y")
    tree = numpy.concatenate(
        orthopy.quadrilateral.tree(numpy.array([x, y]), n, symbolic=True)
    )

    for val in tree:
        assert sympy.integrate(val ** 2, (x, -1, +1), (y, -1, +1)) == 1
    return


def test_show(n=2, r=1):
    def f(X):
        return orthopy.quadrilateral.tree(X, n)[n][r]

    orthopy.quadrilateral.show(f)
    # orthopy.quadrilateral.plot(f)
    # import matplotlib.pyplot as plt
    # plt.savefig('quad.png', transparent=True)
    return


if __name__ == "__main__":
    test_show()
