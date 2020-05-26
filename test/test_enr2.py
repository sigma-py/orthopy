import itertools

import numpy
import sympy
from sympy import oo

import orthopy

standardization = "physicist"


def _integrate(f, X):
    ranges = [(x, -oo, +oo) for x in X]
    return sympy.integrate(f * sympy.exp(-(sum(x ** 2 for x in X))), *ranges)


# def _integrate_monomial(exponents):
#     if any(k % 2 == 1 for k in exponents):
#         return 0
#     return prod(sympy.Rational(2, k + 1) for k in exponents)
#
#
# def _integrate_poly(p):
#     return sum(c * _integrate_monomial(k) for c, k in zip(p.coeffs(), p.monoms()))


def test_integral0(n=3, d=5):
    X = [sympy.symbols("x{}".format(k)) for k in range(d)]
    vals = numpy.concatenate(orthopy.enr2.tree(n, X, standardization, symbolic=True))

    assert _integrate(vals[0], X) == sympy.sqrt(sympy.sqrt(sympy.pi)) ** d
    for val in vals[1:]:
        assert _integrate(val, X) == 0


def test_orthogonality(n=3, d=5):
    X = [sympy.symbols("x{}".format(k)) for k in range(d)]
    tree = numpy.concatenate(orthopy.enr2.tree(n, X, standardization, symbolic=True))
    vals = tree * numpy.roll(tree, 1, axis=0)

    for val in vals:
        assert _integrate(val, X) == 0


def test_normality(n=3, d=5):
    X = [sympy.symbols("x{}".format(k)) for k in range(d)]
    iterator = orthopy.enr2.Iterator(X, standardization, symbolic=True)
    for level in itertools.islice(iterator, n):
        for val in level:
            assert _integrate(val ** 2, X) == 1


if __name__ == "__main__":
    test_normality()
