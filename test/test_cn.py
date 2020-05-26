import math
import operator
import sys
from functools import reduce

import numpy
import sympy

import orthopy

# def _integrate(f, X):
#     integration_limits = [(x, -1, +1) for x in X]
#     return sympy.integrate(f, *integration_limits)


def _prod(iterable):
    if sys.version < "3.8":
        return reduce(operator.mul, iterable, 1)
    return math.prod(iterable)


def _integrate_monomial(exponents):
    if any(k % 2 == 1 for k in exponents):
        return 0
    return _prod(sympy.Rational(2, k + 1) for k in exponents)


def _integrate_poly(p):
    return sum(c * _integrate_monomial(k) for c, k in zip(p.coeffs(), p.monoms()))


def test_integral0(n=4, dim=5):
    """Make sure that the polynomials are orthonormal
    """
    X = [sympy.Symbol("x{}".format(k)) for k in range(dim)]
    p = [sympy.poly(x, X) for x in X]
    vals = numpy.concatenate(orthopy.cn.tree(n, p, symbolic=True))
    vals[0] = sympy.poly(vals[0], X)

    assert _integrate_poly(vals[0]) == sympy.sqrt(2) ** dim
    for val in vals[1:]:
        assert _integrate_poly(val) == 0


def test_orthogonality(n=4, dim=5):
    X = [sympy.Symbol("x{}".format(k)) for k in range(dim)]
    p = [sympy.poly(x, X) for x in X]

    tree = numpy.concatenate(orthopy.cn.tree(n, p, symbolic=True))
    vals = tree * numpy.roll(tree, 1, axis=0)

    for val in vals:
        assert _integrate_poly(val) == 0


def test_normality(n=4, dim=5):
    X = [sympy.Symbol("x{}".format(k)) for k in range(dim)]
    p = [sympy.poly(x, X) for x in X]

    vals = numpy.concatenate(orthopy.cn.tree(n, p, symbolic=True))
    vals[0] = sympy.poly(vals[0], X)

    for val in vals:
        assert _integrate_poly(val ** 2) == 1


if __name__ == "__main__":
    test_integral0()
