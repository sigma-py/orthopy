# -*- coding: utf-8 -*-
#
import numpy
import pytest
import sympy
from sympy import oo

import orthopy


@pytest.mark.parametrize('alpha', [0, 1])
def test_integral0(alpha, n=4):
    '''Make sure that the polynomials are orthonormal
    '''
    x = sympy.Symbol('x')
    vals = numpy.concatenate(
        orthopy.e1r.tree(n, numpy.array([x]), alpha=alpha, symbolic=True)
        )

    assert sympy.integrate(
        vals[0] * x**alpha * sympy.exp(-x), (x, 0, +oo)
        ) == 1

    for val in vals[1:]:
        assert sympy.integrate(
            val * x**alpha * sympy.exp(-x), (x, 0, +oo)
            ) == 0
    return


@pytest.mark.parametrize('alpha', [0, 1])
def test_orthogonality(alpha, n=4):
    x = sympy.Symbol('x')
    tree = numpy.concatenate(
        orthopy.e1r.tree(n, numpy.array([x]), alpha=alpha, symbolic=True)
        )
    vals = tree * numpy.roll(tree, 1, axis=0)

    for val in vals:
        assert sympy.integrate(
            val * x**alpha * sympy.exp(-x), (x, 0, +oo)
            ) == 0
    return


@pytest.mark.parametrize('alpha', [0, 1])
def test_normality(alpha, n=4):
    x = sympy.Symbol('x')
    tree = numpy.concatenate(
        orthopy.e1r.tree(n, numpy.array([x]), alpha=alpha, symbolic=True)
        )

    for val in tree:
        assert sympy.integrate(
            val**2 * x**alpha * sympy.exp(-x), (x, 0, +oo)
            ) == 1
    return


def test_plot():
    orthopy.e1r.plot(L=4)
    return


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    test_plot()
    # plt.show()
    plt.savefig('e1r.png', transparent=True)
