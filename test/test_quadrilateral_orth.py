# -*- coding: utf-8 -*-
#
from __future__ import division

import numpy
import sympy

import orthopy


def test_integral0(n=4, tol=1.0e-14):
    '''Make sure that the polynomials are orthonormal
    '''
    x = sympy.Symbol('x')
    y = sympy.Symbol('y')
    vals = numpy.concatenate(
            orthopy.quadrilateral.tree(n, numpy.array([x, y]), symbolic=True)
            )

    assert sympy.integrate(vals[0], (x, -1, +1), (y, -1, +1)) == 2
    for val in vals[1:]:
        assert sympy.integrate(val, (x, -1, +1), (y, -1, +1)) == 0
    return


def test_orthogonality(n=4, tol=1.0e-14):
    x = sympy.Symbol('x')
    y = sympy.Symbol('y')
    tree = numpy.concatenate(
            orthopy.quadrilateral.tree(n, numpy.array([x, y]), symbolic=True)
            )
    vals = tree * numpy.roll(tree, 1, axis=0)

    for val in vals:
        assert sympy.integrate(val, (x, -1, +1), (y, -1, +1)) == 0
    return


def test_normality(n=4, tol=1.0e-14):
    x = sympy.Symbol('x')
    y = sympy.Symbol('y')
    tree = numpy.concatenate(
            orthopy.quadrilateral.tree(n, numpy.array([x, y]), symbolic=True)
            )

    for val in tree:
        assert sympy.integrate(val**2, (x, -1, +1), (y, -1, +1)) == 1
    return


def test_show(n=2, r=1):
    def f(X):
        return orthopy.quadrilateral.tree(n, X)[n][r]

    orthopy.quadrilateral.show(f)
    # orthopy.quadrilateral.plot(f)
    # import matplotlib.pyplot as plt
    # plt.savefig('quad.png', transparent=True)
    return


if __name__ == '__main__':
    test_show()
