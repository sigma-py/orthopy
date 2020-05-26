import numpy
import sympy

import orthopy
from helpers import prod

X = sympy.symbols("x, y, z")
p = [sympy.poly(x, X) for x in X]


# def _integrate(f, x, y, z):
#     return sympy.integrate(f, (x, -1, +1), (y, -1, +1), (z, -1, +1))


def _integrate_monomial(exponents):
    if any(k % 2 == 1 for k in exponents):
        return 0
    return prod(sympy.Rational(2, k + 1) for k in exponents)


def _integrate_poly(p):
    return sum(c * _integrate_monomial(k) for c, k in zip(p.coeffs(), p.monoms()))


def test_integral0(n=4):
    """Make sure that the polynomials are orthonormal
    """
    vals = numpy.concatenate(orthopy.c3.tree(n, p, symbolic=True))
    vals[0] = sympy.poly(vals[0], X)

    assert _integrate_poly(vals[0]) == sympy.sqrt(2) ** 3
    for val in vals[1:]:
        assert _integrate_poly(val) == 0


def test_orthogonality(n=4):
    tree = numpy.concatenate(orthopy.c3.tree(n, p, symbolic=True))

    vals = tree * numpy.roll(tree, 1, axis=0)
    for val in vals:
        assert _integrate_poly(val) == 0


def test_normality(n=4):
    vals = numpy.concatenate(orthopy.c3.tree(n, p, symbolic=True))
    vals[0] = sympy.poly(vals[0], X)

    for val in vals:
        assert _integrate_poly(val ** 2) == 1


def test_write():
    orthopy.c3.write("hexa.vtk", lambda X: orthopy.c3.tree(5, X.T)[5][5])


if __name__ == "__main__":
    test_write()
