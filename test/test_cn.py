import numpy
import sympy

import orthopy


def test_integral0(n=4, dim=5):
    """Make sure that the polynomials are orthonormal
    """
    variables = numpy.array([sympy.Symbol("x{}".format(k)) for k in range(dim)])
    vals = numpy.concatenate(orthopy.cn.tree(n, variables, symbolic=True))

    integration_limits = [(variable, -1, +1) for variable in variables]
    assert sympy.integrate(vals[0], *integration_limits) == sympy.sqrt(2) ** dim
    for val in vals[1:]:
        assert sympy.integrate(val, *integration_limits) == 0


def test_orthogonality(n=3, dim=5):
    variables = numpy.array([sympy.Symbol("x{}".format(k)) for k in range(dim)])

    tree = numpy.concatenate(orthopy.cn.tree(n, variables, symbolic=True))
    vals = tree * numpy.roll(tree, 1, axis=0)

    integration_limits = [(variable, -1, +1) for variable in variables]
    for val in vals:
        assert sympy.integrate(val, *integration_limits) == 0


def test_normality(n=3, dim=5):
    variables = numpy.array([sympy.Symbol("x{}".format(k)) for k in range(dim)])

    tree = numpy.concatenate(orthopy.cn.tree(n, variables, symbolic=True))

    integration_limits = [(variable, -1, +1) for variable in variables]
    for val in tree:
        assert sympy.integrate(val ** 2, *integration_limits) == 1


if __name__ == "__main__":
    test_integral0()
