import itertools

import numpy
import pytest
import sympy

import ndim
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
    X = [sympy.symbols("x{}".format(k)) for k in range(d)]
    p = [sympy.poly(x, X) for x in X]
    vals = numpy.concatenate(orthopy.enr2.tree(n, p, standardization, symbolic=True))

    assert _integrate_poly(vals[0]) == sympy.sqrt(sympy.sqrt(sympy.pi)) ** d
    for val in vals[1:]:
        assert _integrate_poly(val) == 0


@pytest.mark.parametrize("d,n", [(2, 4), (3, 4), (5, 3)])
def test_orthogonality(d, n):
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
