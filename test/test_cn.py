import numpy
import sympy

import orthopy


def _integrate(f, X):
    integration_limits = [(x, -1, +1) for x in X]
    return sympy.integrate(f, *integration_limits)


def test_integral0(n=4, dim=5):
    """Make sure that the polynomials are orthonormal
    """
    X = [sympy.Symbol("x{}".format(k)) for k in range(dim)]
    vals = numpy.concatenate(orthopy.cn.tree(n, X, symbolic=True))

    assert _integrate(vals[0], X) == sympy.sqrt(2) ** dim
    for val in vals[1:]:
        assert _integrate(val, X) == 0


def test_orthogonality(n=3, dim=5):
    X = [sympy.Symbol("x{}".format(k)) for k in range(dim)]

    tree = numpy.concatenate(orthopy.cn.tree(n, X, symbolic=True))
    vals = tree * numpy.roll(tree, 1, axis=0)

    for val in vals:
        assert _integrate(val, X) == 0


def test_normality(n=3, dim=5):
    X = [sympy.Symbol("x{}".format(k)) for k in range(dim)]

    tree = numpy.concatenate(orthopy.cn.tree(n, X, symbolic=True))

    for val in tree:
        assert _integrate(val ** 2, X) == 1


if __name__ == "__main__":
    test_integral0()
