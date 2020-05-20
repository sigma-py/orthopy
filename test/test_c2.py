import numpy
import sympy

import orthopy


def _integrate(f, x, y):
    return sympy.integrate(f, (x, -1, +1), (y, -1, +1))


def test_integral0(n=4):
    xy = sympy.symbols("x, y")
    vals = numpy.concatenate(orthopy.c2.tree(n, xy, symbolic=True))

    assert _integrate(vals[0], *xy) == 2
    for val in vals[1:]:
        assert _integrate(val, *xy) == 0


def test_orthogonality(n=4):
    xy = sympy.symbols("x, y")
    tree = numpy.concatenate(orthopy.c2.tree(n, xy, symbolic=True))
    vals = tree * numpy.roll(tree, 1, axis=0)

    for val in vals:
        assert _integrate(val, *xy) == 0


def test_normality(n=4):
    xy = sympy.symbols("x, y")
    tree = numpy.concatenate(orthopy.c2.tree(n, xy, symbolic=True))

    for val in tree:
        assert _integrate(val ** 2, *xy) == 1


def test_show(n=2, r=1):
    def f(X):
        return orthopy.c2.tree(n, X)[n][r]

    orthopy.c2.show(f)
    # orthopy.c2.plot(f)
    # import matplotlib.pyplot as plt
    # plt.savefig('quad.png', transparent=True)


if __name__ == "__main__":
    test_show()
