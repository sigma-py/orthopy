# -*- coding: utf-8 -*-
#
from __future__ import division

import numpy
import orthopy
import quadpy


def test_integral0(n=4, tol=1.0e-14):
    '''Make sure that the polynomials are orthonormal
    '''
    def ff(x):
        return numpy.concatenate(
            orthopy.quadrilateral.orth_tree(n, x, symbolic=False)
            )

    quad = [[[-1.0, -1.0], [+1.0, -1.0]], [[-1.0, +1.0], [+1.0, +1.0]]]
    scheme = quadpy.quadrilateral.Dunavant(4)
    val = quadpy.quadrilateral.integrate(ff, quad, scheme)
    assert numpy.all(abs(val[0] - 2.0) < tol)
    assert numpy.all(abs(val[1:]) < tol)
    return


def test_orthogonality(n=4, tol=1.0e-14):
    def f_shift(x):
        tree = numpy.concatenate(
                orthopy.quadrilateral.orth_tree(n, x)
                )
        return tree * numpy.roll(tree, 1, axis=0)

    quad = [[[-1.0, -1.0], [+1.0, -1.0]], [[-1.0, +1.0], [+1.0, +1.0]]]
    scheme = quadpy.quadrilateral.Dunavant(4)
    val = quadpy.quadrilateral.integrate(f_shift, quad, scheme)
    assert numpy.all(abs(val) < tol)
    return


def test_normality(n=4, tol=1.0e-14):
    def ff(x):
        tree = numpy.concatenate(
                orthopy.quadrilateral.orth_tree(n, x)
                )
        return tree * tree

    quad = [[[-1.0, -1.0], [+1.0, -1.0]], [[-1.0, +1.0], [+1.0, +1.0]]]
    scheme = quadpy.quadrilateral.Dunavant(4)
    assert scheme.degree >= 2*n
    vals = quadpy.quadrilateral.integrate(ff, quad, scheme)
    assert numpy.all(abs(vals - 1) < tol)
    exit(1)
    return


# TODO
# def test_show(n=2, r=1):
#     def f(X):
#         return orthopy.quadrilateral.orth_tree(n, X)[n][r]
#
#     val = orthopy.quadrilateral.orth_tree(
#         2, numpy.array([[+1], [0]], dtype=int)
#         )[1][0]
#     print(val)
#     exit(1)
#
#     orthopy.quadrilateral.show(f)
#     return


if __name__ == '__main__':
    test_integral0()
    test_orthogonality()
    test_normality()
