import numpy
import sympy

import orthopy


def _integrate(f, x, y):
    # Cartesian integration in sympy is bugged, cf.
    # <https://github.com/sympy/sympy/issues/13816>.
    # Simply transform to polar coordinates for now.
    r = sympy.Symbol("r")
    phi = sympy.Symbol("phi")
    return sympy.integrate(
        r * f.subs([(x, r * sympy.cos(phi)), (y, r * sympy.sin(phi))]),
        (r, 0, 1),
        (phi, 0, 2 * sympy.pi),
    )


def test_integral0(n=4):
    xy = sympy.symbols("x, y")
    vals = numpy.concatenate(orthopy.s2.tree(xy, n, symbolic=True))

    assert _integrate(vals[0], *xy) == sympy.sqrt(sympy.pi)
    for val in vals[1:]:
        assert _integrate(val, *xy) == 0


def test_orthogonality(n=3):
    xy = sympy.symbols("x, y")
    tree = numpy.concatenate(orthopy.s2.tree(xy, n, symbolic=True))
    vals = tree * numpy.roll(tree, 1, axis=0)

    for val in vals:
        assert _integrate(val, *xy) == 0


def test_normality(n=4):
    xy = sympy.symbols("x, y")
    for k, level in enumerate(orthopy.s2.Iterator(xy, symbolic=True)):
        for val in level:
            assert _integrate(val ** 2, *xy) == 1
        if k == n:
            break


def test_show(n=2, r=1):
    def f(X):
        return orthopy.s2.tree(X, n)[n][r]

    orthopy.s2.show(f)
    # orthopy.s2.plot(f, lcar=2.0e-2)
    # import matplotlib.pyplot as plt
    # plt.savefig('s2.png', transparent=True)


if __name__ == "__main__":
    test_show()
