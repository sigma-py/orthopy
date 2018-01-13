# -*- coding: utf-8 -*-
#
from __future__ import division, print_function

import numpy
import sympy

import orthopy


def test_integral0(n=4):
    '''Make sure that the polynomials are orthonormal
    '''
    x = sympy.Symbol('x')
    y = sympy.Symbol('y')
    vals = numpy.concatenate(
        orthopy.disk.tree(numpy.array([x, y]), n, symbolic=True)
        )

    # Cartesian integration in sympy is bugged, cf.
    # <https://github.com/sympy/sympy/issues/13816>.
    # Simply transform to polar coordinates for now.
    r = sympy.Symbol('r')
    phi = sympy.Symbol('phi')
    assert sympy.integrate(
        r * vals[0].subs([(x, r*sympy.cos(phi)), (y, r*sympy.sin(phi))]),
        (r, 0, 1), (phi, 0, 2*sympy.pi)
        ) == sympy.sqrt(sympy.pi)

    for val in vals[1:]:
        assert sympy.integrate(
            r * val.subs([(x, r*sympy.cos(phi)), (y, r*sympy.sin(phi))]),
            (r, 0, 1), (phi, 0, 2*sympy.pi)
            ) == 0

    return


def test_orthogonality(n=3):
    x = sympy.Symbol('x')
    y = sympy.Symbol('y')
    tree = numpy.concatenate(
        orthopy.disk.tree(numpy.array([x, y]), n, symbolic=True)
        )
    vals = tree * numpy.roll(tree, 1, axis=0)

    r = sympy.Symbol('r')
    phi = sympy.Symbol('phi')
    for val in vals:
        assert sympy.integrate(
            r * val.subs([(x, r*sympy.cos(phi)), (y, r*sympy.sin(phi))]),
            (r, 0, 1), (phi, 0, 2*sympy.pi)
            ) == 0
    return


def test_normality(n=4):
    x = sympy.Symbol('x')
    y = sympy.Symbol('y')
    tree = numpy.concatenate(
        orthopy.disk.tree(numpy.array([x, y]), n, symbolic=True)
        )

    # Cartesian integration in sympy is bugged, cf.
    # <https://github.com/sympy/sympy/issues/13816>.
    # Simply transform to polar coordinates for now.
    r = sympy.Symbol('r')
    phi = sympy.Symbol('phi')
    for val in tree:
        assert sympy.integrate(
            r * val.subs([(x, r*sympy.cos(phi)), (y, r*sympy.sin(phi))])**2,
            (r, 0, 1), (phi, 0, 2*sympy.pi)
            ) == 1
    return


def test_show(n=4, r=3):
    def f(X):
        return orthopy.disk.tree(X, n)[n][r]

    orthopy.disk.show(f)
    # orthopy.disk.plot(f, lcar=2.0e-2)
    # import matplotlib.pyplot as plt
    # plt.savefig('disk.png', transparent=True)
    return


if __name__ == '__main__':
    test_show()
