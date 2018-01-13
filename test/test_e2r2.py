# -*- coding: utf-8 -*-
#
from __future__ import division

import numpy
import sympy
from sympy import oo

import orthopy


def test_integral0(n=4):
    '''Make sure that the polynomials are orthonormal
    '''
    x = sympy.Symbol('x')
    y = sympy.Symbol('y')
    vals = numpy.concatenate(
        orthopy.e2r2.tree(numpy.array([x, y]), n, symbolic=True)
        )

    assert sympy.integrate(
        vals[0] * sympy.exp(-x**2-y**2), (x, -oo, +oo), (y, -oo, +oo)
        ) == sympy.sqrt(sympy.pi)
    for val in vals[1:]:
        assert sympy.integrate(
            val * sympy.exp(-x**2-y**2), (x, -oo, +oo), (y, -oo, +oo)
            ) == 0
    return


def test_orthogonality(n=4):
    x = sympy.Symbol('x')
    y = sympy.Symbol('y')
    tree = numpy.concatenate(
        orthopy.e2r2.tree(numpy.array([x, y]), n, symbolic=True)
        )
    vals = tree * numpy.roll(tree, 1, axis=0)

    for val in vals:
        assert sympy.integrate(
            val * sympy.exp(-x**2 - y**2), (x, -oo, +oo), (y, -oo, +oo)
            ) == 0
    return


def test_normality(n=4):
    x = sympy.Symbol('x')
    y = sympy.Symbol('y')
    tree = numpy.concatenate(
        orthopy.e2r2.tree(numpy.array([x, y]), n, symbolic=True)
        )

    for val in tree:
        assert sympy.integrate(
            val**2 * sympy.exp(-x**2 - y**2), (x, -oo, +oo), (y, -oo, +oo)
            ) == 1
    return


def test_show(n=6, r=3, d=1.5):
    def f(X):
        return orthopy.e2r2.tree(X, n)[n][r]

    orthopy.e2r2.show(f, d=d)
    # orthopy.e2r2.plot(f)
    # import matplotlib.pyplot as plt
    # plt.savefig('e2r2.png', transparent=True)
    return


if __name__ == '__main__':
    test_show(n=6, r=3)
