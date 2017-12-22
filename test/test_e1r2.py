# -*- coding: utf-8 -*-
#
import numpy
import sympy
from sympy import oo

import orthopy


def test_integral0(n=4):
    '''Make sure that the polynomials are orthonormal
    '''
    x = sympy.Symbol('x')
    vals = numpy.concatenate(
        orthopy.e1r2.tree(n, numpy.array([x]), symbolic=True)
        )

    assert sympy.integrate(
        vals[0] * sympy.exp(-x**2), (x, -oo, +oo)
        ) == sympy.sqrt(sympy.sqrt(sympy.pi))

    for k, val in enumerate(vals[1:]):
        assert sympy.integrate(
            val * sympy.exp(-x**2), (x, -oo, +oo)
            ) == 0
    return


def test_orthogonality(n=4):
    x = sympy.Symbol('x')
    tree = numpy.concatenate(
        orthopy.e1r2.tree(n, numpy.array([x]), symbolic=True)
        )
    vals = tree * numpy.roll(tree, 1, axis=0)

    for val in vals:
        assert sympy.integrate(val * sympy.exp(-x**2), (x, -oo, +oo)) == 0
    return


def test_normality(n=4):
    x = sympy.Symbol('x')
    tree = numpy.concatenate(
        orthopy.e1r2.tree(n, numpy.array([x]), symbolic=True)
        )

    for k, val in enumerate(tree):
        assert sympy.integrate(
            val**2 * sympy.exp(-x**2), (x, -oo, +oo)
            ) == 1
    return


def test_plot():
    orthopy.e1r2.plot(L=4)
    return


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    test_plot()
    plt.show()
    # plt.savefig('e1r2.png', transparent=True)
