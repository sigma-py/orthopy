import itertools

import numpy
import pytest
import sympy
from helpers_s2 import _integrate_poly

import orthopy

X = sympy.symbols("x, y")
P = [sympy.poly(x, X) for x in X]


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
    iterator = orthopy.s2.yu.Eval(P, scaling, symbolic=True)
    for k, vals in enumerate(itertools.islice(iterator, n)):
        if k == 0:
            assert _integrate_poly(vals[0]) == int0
        else:
            for val in vals[1:]:
                assert _integrate_poly(val) == 0


@pytest.mark.parametrize("scaling", ["classical", "monic", "normal"])
def test_yu_orthogonality(scaling, n=4):
    evaluator = orthopy.s2.yu.Eval(P, scaling, symbolic=True)
    vals = numpy.concatenate([next(evaluator) for _ in range(n + 1)])
    for f0, f1 in itertools.combinations(vals, 2):
        assert _integrate_poly(f0 * f1) == 0


def test_yu_normality(n=4):
    iterator = orthopy.s2.yu.Eval(P, "normal", symbolic=True)
    for vals in itertools.islice(iterator, n):
        for val in vals:
            assert _integrate_poly(val ** 2) == 1


@pytest.mark.parametrize("degrees", [(2, 1)])
def test_show(degrees, scaling="normal"):
    orthopy.s2.yu.show_single(degrees, scaling=scaling)
    orthopy.s2.yu.savefig_single("disk-yu.png", degrees, scaling=scaling)


@pytest.mark.parametrize("n", [2])
def test_show_tree(n, scaling="normal"):
    orthopy.s2.yu.show_tree(n, scaling=scaling, colorbar=True, clim=(-1.7, 1.7))
    orthopy.s2.yu.savefig_tree(
        "disk-yu-tree.png", n, scaling=scaling, colorbar=True, clim=(-1.7, 1.7)
    )


if __name__ == "__main__":
    # test_show((3, 2), "normal")
    test_show_tree(5, "normal")
