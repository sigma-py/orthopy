import itertools

import numpy
import pytest
import sympy

import orthopy
from helpers import prod

# def _integrate(f, X):
#     integration_limits = [(x, -1, +1) for x in X]
#     return sympy.integrate(f, *integration_limits)


def _integrate_monomial(exponents):
    if any(k % 2 == 1 for k in exponents):
        return 0
    return prod(sympy.Rational(2, k + 1) for k in exponents)


def _integrate_poly(p):
    return sum(c * _integrate_monomial(k) for c, k in zip(p.coeffs(), p.monoms()))


@pytest.mark.parametrize("d", [2, 3, 5])
def test_integral0(d, n=4):
    """Make sure that the polynomials are orthonormal
    """
    X = [sympy.Symbol("x{}".format(k)) for k in range(d)]
    p = [sympy.poly(x, X) for x in X]
    vals = numpy.concatenate(orthopy.cn.tree(n, p, symbolic=True))
    vals[0] = sympy.poly(vals[0], X)

    assert _integrate_poly(vals[0]) == sympy.sqrt(2) ** d
    for val in vals[1:]:
        assert _integrate_poly(val) == 0


@pytest.mark.parametrize("d", [2, 3, 5])
def test_orthogonality(d, n=4):
    X = [sympy.Symbol("x{}".format(k)) for k in range(d)]
    p = [sympy.poly(x, X) for x in X]
    tree = numpy.concatenate(orthopy.cn.tree(n, p, symbolic=True))
    for f0, f1 in itertools.combinations(tree, 2):
        assert _integrate_poly(f0 * f1) == 0


@pytest.mark.parametrize("d", [2, 3, 5])
def test_normality(d, n=4):
    X = [sympy.Symbol("x{}".format(k)) for k in range(d)]
    p = [sympy.poly(x, X) for x in X]

    vals = numpy.concatenate(orthopy.cn.tree(n, p, symbolic=True))
    vals[0] = sympy.poly(vals[0], X)

    for val in vals:
        assert _integrate_poly(val ** 2) == 1


if __name__ == "__main__":
    test_integral0()
