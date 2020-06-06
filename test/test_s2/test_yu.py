import itertools

import numpy
import pytest
import sympy

import orthopy
from helpers_s2 import _integrate_poly

X = sympy.symbols("x, y")


# def _integrate(f):
#     # Cartesian integration in sympy is bugged, cf.
#     # <https://github.com/sympy/sympy/issues/13816>.
#     # Simply transform to polar coordinates for now.
#     r = sympy.Symbol("r")
#     phi = sympy.Symbol("phi")
#     return sympy.integrate(
#         r * f.subs([(X[0], r * sympy.cos(phi)), (X[1], r * sympy.sin(phi))]),
#         (r, 0, 1),
#         (phi, 0, 2 * sympy.pi),
#     )


@pytest.mark.parametrize(
    "scaling,int0",
    [("classical", sympy.pi), ("monic", sympy.pi), ("normal", sympy.sqrt(sympy.pi))],
)
def test_yu_integral0(scaling, int0, n=4):
    p = [sympy.poly(x, X) for x in X]
    vals = numpy.concatenate(orthopy.s2.yu.tree(n, p, scaling, symbolic=True))
    vals[0] = sympy.poly(vals[0], X)

    assert _integrate_poly(vals[0]) == int0
    for val in vals[1:]:
        assert _integrate_poly(val) == 0


@pytest.mark.parametrize("scaling", ["classical", "monic", "normal"])
def test_yu_orthogonality(scaling, n=4):
    p = [sympy.poly(x, X) for x in X]
    tree = numpy.concatenate(orthopy.s2.yu.tree(n, p, scaling, symbolic=True))
    vals = tree * numpy.roll(tree, 1, axis=0)

    for val in vals:
        assert _integrate_poly(val) == 0


def test_yu_normality(n=4):
    p = [sympy.poly(x, X) for x in X]
    iterator = orthopy.s2.yu.Eval(p, "normal", symbolic=True)
    for k, vals in enumerate(itertools.islice(iterator, n)):
        if k == 0:
            vals[0] = sympy.poly(vals[0], X)
        for val in vals:
            assert _integrate_poly(val ** 2) == 1


def test_show(scaling="normal", n=2, r=1):
    def f(X):
        return orthopy.s2.zernike.tree(n, X, scaling)[n][r]

    # k = numpy.linspace(0, 2 * numpy.pi, 100000)
    # X = numpy.array([numpy.cos(k), numpy.sin(k)])
    # print(numpy.min(f(X)), numpy.max(f(X)))
    # x0 = [numpy.sqrt(1 / (n)), numpy.sqrt((n-1) / (n))]
    # print(f(x0))

    # p = [sympy.poly(x, X) for x in X]
    # iterator = orthopy.s2.yu.Eval(p, "classical", symbolic=True)
    # for k, vals in enumerate(itertools.islice(iterator, 10)):
    #     print()
    #     for val in vals:
    #         print(val)
    # exit(1)

    orthopy.s2.show(f, lcar=1.0e-2)
    # orthopy.s2.plot(f, lcar=2.0e-2)
    # import matplotlib.pyplot as plt
    # plt.savefig('s2.png', transparent=True)


if __name__ == "__main__":
    # p = [sympy.poly(x, X) for x in X]
    # X = [1.0, 0.0]
    # iterator = orthopy.s2.zernike.Eval(X, "classical", symbolic=False)
    # for vals in itertools.islice(iterator, 10):
    #     print(vals[0])
    # test_zernike()
    test_show("classical", n=5, r=2)
