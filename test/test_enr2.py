import itertools

import numpy
import pytest
import sympy
from sympy import Rational, pi, sqrt

import orthopy

standardization = "physicist"


# def _integrate(f, X):
#     ranges = [(x, -oo, +oo) for x in X]
#     return sympy.integrate(f * sympy.exp(-(sum(x ** 2 for x in X))), *ranges)


def _integrate_monomial(exponents):
    if any(k % 2 == 1 for k in exponents):
        return 0

    if all(k == 0 for k in exponents):
        n = len(exponents)
        return sqrt(pi) ** n

    # find first nonzero
    idx = next(i for i, j in enumerate(exponents) if j > 0)
    alpha = Rational(exponents[idx] - 1, 2)
    k2 = exponents.copy()
    k2[idx] -= 2
    return _integrate_monomial(k2) * alpha


def _integrate_poly(p):
    return sum(c * _integrate_monomial(list(k)) for c, k in zip(p.coeffs(), p.monoms()))


@pytest.mark.parametrize("d", [2, 3, 5])
def test_integral0(d, n=4):
    X = [sympy.symbols("x{}".format(k)) for k in range(d)]
    p = [sympy.poly(x, X) for x in X]
    vals = numpy.concatenate(orthopy.enr2.tree(n, p, standardization, symbolic=True))
    vals[0] = sympy.poly(vals[0], X)

    assert _integrate_poly(vals[0]) == sympy.sqrt(sympy.sqrt(sympy.pi)) ** d
    for val in vals[1:]:
        assert _integrate_poly(val) == 0


@pytest.mark.parametrize("d", [2, 3, 5])
def test_orthogonality(d, n=4):
    X = [sympy.symbols("x{}".format(k)) for k in range(d)]
    p = [sympy.poly(x, X) for x in X]
    tree = numpy.concatenate(orthopy.enr2.tree(n, p, standardization, symbolic=True))
    for f0, f1 in itertools.combinations(tree, 2):
        assert _integrate_poly(f0 * f1) == 0


@pytest.mark.parametrize("d", [2, 3, 5])
def test_normality(d, n=4):
    X = [sympy.symbols("x{}".format(k)) for k in range(d)]
    p = [sympy.poly(x, X) for x in X]
    iterator = orthopy.enr2.Eval(p, standardization, symbolic=True)

    for k, level in enumerate(itertools.islice(iterator, n)):
        if k == 0:
            level[0] = sympy.poly(level[0], X)
        for val in level:
            assert _integrate_poly(val ** 2) == 1


if __name__ == "__main__":
    test_normality()
