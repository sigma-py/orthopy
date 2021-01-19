import itertools

import ndim
import numpy as np
import pytest
import sympy

import orthopy

# def _integrate(f, X):
#     ranges = [(x, -oo, +oo) for x in X]
#     return sympy.integrate(f * sympy.exp(-(sum(x ** 2 for x in X))), *ranges)


def _integrate_poly(p, standardization):
    return sum(
        c * ndim.enr2.integrate_monomial(k, standardization, symbolic=True)
        for c, k in zip(p.coeffs(), p.monoms())
    )


@pytest.mark.parametrize("d", [2, 3, 5])
@pytest.mark.parametrize("standardization", ["physicists", "probabilists"])
def test_integral0(d, standardization, n=4):
    X = [sympy.symbols(f"x{k}") for k in range(d)]
    p = [sympy.poly(x, X) for x in X]
    evaluator = orthopy.enr2.Eval(p, standardization)

    vals = next(evaluator)
    ref = (
        sympy.sqrt(sympy.sqrt(sympy.pi)) ** d if standardization == "physicists" else 1
    )
    assert _integrate_poly(vals[0], standardization) == ref
    for k in range(n):
        vals = next(evaluator)
        for val in vals:
            assert _integrate_poly(val, standardization) == 0


@pytest.mark.parametrize("d,n", [(2, 4), (3, 4), (5, 3)])
@pytest.mark.parametrize("standardization", ["physicists", "probabilists"])
def test_orthogonality(d, n, standardization):
    X = [sympy.symbols(f"x{k}") for k in range(d)]
    p = [sympy.poly(x, X) for x in X]
    evaluator = orthopy.enr2.Eval(p, standardization)
    vals = np.concatenate([next(evaluator) for _ in range(n + 1)])
    for f0, f1 in itertools.combinations(vals, 2):
        assert _integrate_poly(f0 * f1, standardization) == 0


@pytest.mark.parametrize("d", [2, 3, 5])
@pytest.mark.parametrize("standardization", ["physicists", "probabilists"])
def test_normality(d, standardization, n=4):
    X = [sympy.symbols(f"x{k}") for k in range(d)]
    p = [sympy.poly(x, X) for x in X]
    evaluator = orthopy.enr2.Eval(p, standardization)

    vals = next(evaluator)
    val = sympy.poly(vals[0], X)
    assert _integrate_poly(val ** 2, standardization) == 1

    for k in range(n + 1):
        vals = next(evaluator)
        for val in vals:
            assert _integrate_poly(val ** 2, standardization) == 1


@pytest.mark.parametrize("n", [2])
def test_show_tree(n):
    standardization = "probabilists"
    # 1D
    orthopy.enr2.show_tree_1d(n, standardization, "normal")
    orthopy.enr2.savefig_tree_1d("e1r2.svg", n, standardization, "normal")
    # 2D
    orthopy.enr2.show_tree_2d(n, standardization, clim=(-2, 2))
    orthopy.enr2.savefig_tree_2d("e2r2.png", n, standardization, clim=(-2, 2))


@pytest.mark.parametrize("n", [2])
def test_write_tree(n):
    standardization = "probabilists"
    # 3D
    orthopy.enr2.write_tree_3d("e3r2.vtk", n, standardization)


if __name__ == "__main__":
    test_show_tree(5)
    # test_write_tree(5)
