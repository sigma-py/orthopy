import itertools

import ndim
import numpy
import pytest
import sympy

import orthopy

standardization = "physicist"


# def _integrate(f, X):
#     ranges = [(x, -oo, +oo) for x in X]
#     return sympy.integrate(f * sympy.exp(-(sum(x ** 2 for x in X))), *ranges)


def _integrate_poly(p):
    return sum(
        c * ndim.enr2.integrate_monomial(k, "physicists", symbolic=True)
        for c, k in zip(p.coeffs(), p.monoms())
    )


@pytest.mark.parametrize("d", [2, 3, 5])
def test_integral0(d, n=4):
    X = [sympy.symbols(f"x{k}") for k in range(d)]
    p = [sympy.poly(x, X) for x in X]
    evaluator = orthopy.enr2.Eval(p, standardization, symbolic=True)

    vals, _ = next(evaluator)
    assert _integrate_poly(vals[0]) == sympy.sqrt(sympy.sqrt(sympy.pi)) ** d
    for k in range(n):
        vals, _ = next(evaluator)
        for val in vals:
            assert _integrate_poly(val) == 0


@pytest.mark.parametrize("d,n", [(2, 4), (3, 4), (5, 3)])
def test_orthogonality(d, n):
    X = [sympy.symbols(f"x{k}") for k in range(d)]
    p = [sympy.poly(x, X) for x in X]
    evaluator = orthopy.enr2.Eval(p, standardization, symbolic=True)
    vals = numpy.concatenate([next(evaluator)[0] for _ in range(n + 1)])
    for f0, f1 in itertools.combinations(vals, 2):
        assert _integrate_poly(f0 * f1) == 0


@pytest.mark.parametrize("d", [2, 3, 5])
def test_normality(d, n=4):
    X = [sympy.symbols(f"x{k}") for k in range(d)]
    p = [sympy.poly(x, X) for x in X]
    evaluator = orthopy.enr2.Eval(p, standardization, symbolic=True)

    vals, _ = next(evaluator)
    val = sympy.poly(vals[0], X)
    assert _integrate_poly(val ** 2) == 1

    for k in range(n + 1):
        vals, _ = next(evaluator)
        for val in vals:
            assert _integrate_poly(val ** 2) == 1


if __name__ == "__main__":
    test_normality()
