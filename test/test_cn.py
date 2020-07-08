import itertools

import ndim
import numpy
import pytest
import sympy

import orthopy

# def _integrate(f, X):
#     integration_limits = [(x, -1, +1) for x in X]
#     return sympy.integrate(f, *integration_limits)


def _integrate_poly(p):
    return sum(
        c * ndim.ncube.integrate_monomial(k, symbolic=True)
        for c, k in zip(p.coeffs(), p.monoms())
    )


@pytest.mark.parametrize("d", [2, 3, 5])
def test_integral0(d, n=4):
    """Make sure that the polynomials are orthonormal
    """
    X = [sympy.Symbol(f"x{k}") for k in range(d)]
    p = [sympy.poly(x, X) for x in X]
    iterator = orthopy.cn.Eval(p, symbolic=True)

    for k, vals in enumerate(itertools.islice(iterator, n)):
        if k == 0:
            assert _integrate_poly(vals[0]) == sympy.sqrt(2) ** d
        else:
            for val in vals:
                assert _integrate_poly(val) == 0


@pytest.mark.parametrize("d,n", [(2, 4), (3, 4), (5, 3)])
def test_orthogonality(d, n):
    X = [sympy.Symbol(f"x{k}") for k in range(d)]
    p = [sympy.poly(x, X) for x in X]
    evaluator = orthopy.cn.Eval(p, symbolic=True)
    tree = numpy.concatenate([next(evaluator) for _ in range(n + 1)])
    for f0, f1 in itertools.combinations(tree, 2):
        assert _integrate_poly(f0 * f1) == 0


@pytest.mark.parametrize("d", [2, 3, 5])
def test_normality(d, n=4):
    X = [sympy.Symbol(f"x{k}") for k in range(d)]
    p = [sympy.poly(x, X) for x in X]

    evaluator = orthopy.cn.Eval(p, symbolic=True)
    vals = numpy.concatenate([next(evaluator) for _ in range(n + 1)])

    for val in vals:
        assert _integrate_poly(val ** 2) == 1


def test_show_tree(n):
    alpha = 0.0
    beta = 0.0

    # 1D
    orthopy.cn.show_tree_1d(n, alpha, beta, "normal")
    orthopy.cn.savefig_tree_1d("c1.svg", n, alpha, beta, "normal")

    # 2D
    orthopy.cn.show_tree_2d(n, alpha, beta, clim=(-1, 1))
    orthopy.cn.savefig_tree_2d("c2.png", n, alpha, beta, clim=(-1, 1))


if __name__ == "__main__":
    test_show_tree(5)
