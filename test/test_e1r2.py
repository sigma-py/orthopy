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
        print(val)
        print(k+1, sympy.simplify(sympy.integrate(
            val * sympy.exp(-x**2), (x, -oo, +oo)
            )))
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
        print(k, sympy.integrate(
            val**2 * sympy.exp(-x**2), (x, -oo, +oo)
            ))
        assert sympy.integrate(
            val**2 * sympy.exp(-x**2), (x, -oo, +oo)
            ) == 1
    return


# def test_write():
#     orthopy.e1r2.write(
#         'hexa.vtu',
#         lambda X: orthopy.e1r2.tree(5, X.T)[5][5]
#         )
#     return


# if __name__ == '__main__':
#     test_write()
