import itertools

import numpy as np
import pytest
import sympy
from helpers_s2 import _integrate_poly

import orthopy

X = sympy.symbols("x, y")
P = [sympy.poly(x, X) for x in X]


def test_zernike2_explicit():
    x, y = X
    ref = [
        [1],
        [x - y, x + y],
        [
            x ** 2 - 2 * x * y - y ** 2,
            2 * (x ** 2 + y ** 2) - 1,
            x ** 2 + 2 * x * y - y ** 2,
        ],
        [
            x ** 3 - 3 * x ** 2 * y - 3 * x * y ** 2 + y ** 3,
            3 * x ** 3 - 3 * x ** 2 * y + 3 * x * y ** 2 - 2 * x - 3 * y ** 3 + 2 * y,
            3 * x ** 3 + 3 * x ** 2 * y + 3 * x * y ** 2 - 2 * x + 3 * y ** 3 - 2 * y,
            x ** 3 + 3 * x ** 2 * y - 3 * x * y ** 2 - y ** 3,
        ],
        [
            x ** 4 - 4 * x ** 3 * y - 6 * x ** 2 * y ** 2 + 4 * x * y ** 3 + y ** 4,
            4 * x ** 4
            - 8 * x ** 3 * y
            - 3 * x ** 2
            - 8 * x * y ** 3
            + 6 * x * y
            - 4 * y ** 4
            + 3 * y ** 2,
            6 * (x ** 2 + y ** 2) ** 2 - 6 * (x ** 2 + y ** 2) + 1,
            4 * x ** 4
            + 8 * x ** 3 * y
            - 3 * x ** 2
            + 8 * x * y ** 3
            - 6 * x * y
            - 4 * y ** 4
            + 3 * y ** 2,
            x ** 4 + 4 * x ** 3 * y - 6 * x ** 2 * y ** 2 - 4 * x * y ** 3 + y ** 4,
        ],
        [
            x ** 5
            - 5 * x ** 4 * y
            - 10 * x ** 3 * y ** 2
            + 10 * x ** 2 * y ** 3
            + 5 * x * y ** 4
            - y ** 5,
            5 * x ** 5
            - 15 * x ** 4 * y
            - 10 * x ** 3 * y ** 2
            - 4 * x ** 3
            - 10 * x ** 2 * y ** 3
            + 12 * x ** 2 * y
            - 15 * x * y ** 4
            + 12 * x * y ** 2
            + 5 * y ** 5
            - 4 * y ** 3,
            10 * x ** 5
            - 10 * x ** 4 * y
            + 20 * x ** 3 * y ** 2
            - 12 * x ** 3
            - 20 * x ** 2 * y ** 3
            + 12 * x ** 2 * y
            + 10 * x * y ** 4
            - 12 * x * y ** 2
            + 3 * x
            - 10 * y ** 5
            + 12 * y ** 3
            - 3 * y,
            10 * x ** 5
            + 10 * x ** 4 * y
            + 20 * x ** 3 * y ** 2
            - 12 * x ** 3
            + 20 * x ** 2 * y ** 3
            - 12 * x ** 2 * y
            + 10 * x * y ** 4
            - 12 * x * y ** 2
            + 3 * x
            + 10 * y ** 5
            - 12 * y ** 3
            + 3 * y,
            5 * x ** 5
            + 15 * x ** 4 * y
            - 10 * x ** 3 * y ** 2
            - 4 * x ** 3
            + 10 * x ** 2 * y ** 3
            - 12 * x ** 2 * y
            - 15 * x * y ** 4
            + 12 * x * y ** 2
            - 5 * y ** 5
            + 4 * y ** 3,
            x ** 5
            + 5 * x ** 4 * y
            - 10 * x ** 3 * y ** 2
            - 10 * x ** 2 * y ** 3
            + 5 * x * y ** 4
            + y ** 5,
        ],
    ]
    iterator = orthopy.s2.zernike2.Eval(X, "classical")
    for ref_level, vals in zip(ref, itertools.islice(iterator, 6)):
        for r, val in zip(ref_level, vals):
            # r = sympy.simplify(r)
            val = sympy.simplify(val)
            assert sympy.simplify(r - val) == 0, f"ref = {r}  !=   {val}"


@pytest.mark.parametrize(
    "scaling,int0",
    [("classical", sympy.pi), ("monic", sympy.pi), ("normal", sympy.sqrt(sympy.pi))],
)
def test_zernike2_integral0(scaling, int0, n=4):
    iterator = orthopy.s2.zernike2.Eval(P, scaling)

    for k, vals in enumerate(itertools.islice(iterator, n)):
        if k == 0:
            assert _integrate_poly(vals[0]) == int0
        else:
            for val in vals:
                assert _integrate_poly(val) == 0


@pytest.mark.parametrize("scaling", ["classical", "monic", "normal"])
def test_zernike2_orthogonality(scaling, n=4):
    evaluator = orthopy.s2.zernike2.Eval(P, scaling)
    vals = np.concatenate([next(evaluator) for _ in range(n + 1)])
    for f0, f1 in itertools.combinations(vals, 2):
        assert _integrate_poly(f0 * f1) == 0


def test_zernike2_normality(n=4):
    iterator = orthopy.s2.zernike2.Eval(P, "normal")
    for k, vals in enumerate(itertools.islice(iterator, n)):
        for val in vals:
            assert _integrate_poly(val ** 2) == 1


@pytest.mark.parametrize("degrees", [(2, 1)])
def test_show(degrees, scaling="normal"):
    orthopy.s2.zernike2.show_single(degrees)


@pytest.mark.parametrize("n", [2])
def test_show_tree(n, scaling="normal"):
    orthopy.s2.zernike2.show_tree(n, scaling=scaling, colorbar=True, clim=(-1.7, 1.7))
    orthopy.s2.zernike2.savefig_tree(
        "disk-zernike2-tree.png", n, scaling=scaling, colorbar=True, clim=(-1.7, 1.7)
    )


if __name__ == "__main__":
    # test_show((3, 2), "normal")
    test_show_tree(5, "normal")
