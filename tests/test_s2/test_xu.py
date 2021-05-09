import itertools

import numpy as np
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
def test_xu_integral0(scaling, int0, n=4):
    iterator = orthopy.s2.xu.Eval(P, scaling)
    for k, vals in enumerate(itertools.islice(iterator, n)):
        if k == 0:
            assert _integrate_poly(vals[0]) == int0
        else:
            for val in vals[1:]:
                assert _integrate_poly(val) == 0


@pytest.mark.parametrize("scaling", ["classical", "monic", "normal"])
def test_xu_orthogonality(scaling, n=4):
    evaluator = orthopy.s2.xu.Eval(P, scaling)
    vals = np.concatenate([next(evaluator) for _ in range(n + 1)])
    for f0, f1 in itertools.combinations(vals, 2):
        assert _integrate_poly(f0 * f1) == 0


def test_xu_normality(n=4):
    iterator = orthopy.s2.xu.Eval(P, "normal")
    for vals in itertools.islice(iterator, n):
        for val in vals:
            assert _integrate_poly(val ** 2) == 1


@pytest.mark.parametrize("degrees", [(2, 1)])
def test_show(degrees, scaling="normal"):
    orthopy.s2.xu.show_single(degrees, scaling=scaling)
    orthopy.s2.xu.savefig_single("disk-xu.png", degrees, scaling=scaling)


@pytest.mark.parametrize("n", [2])
def test_show_tree(n, scaling="normal"):
    orthopy.s2.xu.show_tree(n, scaling=scaling, colorbar=True, clim=(-1.7, 1.7))
    orthopy.s2.xu.savefig_tree(
        "disk-xu-tree.png", n, scaling=scaling, colorbar=True, clim=(-1.7, 1.7)
    )


if __name__ == "__main__":
    # test_show((3, 2), "normal")
    test_show_tree(5, "normal")
