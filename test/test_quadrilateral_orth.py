# -*- coding: utf-8 -*-
#
from __future__ import division

import numpy
import quadpy

import orthopy


def test_integral0(n=4, tol=1.0e-14):
    '''Make sure that the polynomials are orthonormal
    '''
    def ff(x):
        return numpy.concatenate(
            orthopy.quadrilateral.tree(n, x, symbolic=False)
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
                orthopy.quadrilateral.tree(n, x)
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
                orthopy.quadrilateral.tree(n, x)
                )
        return tree * tree

    quad = [[[-1.0, -1.0], [+1.0, -1.0]], [[-1.0, +1.0], [+1.0, +1.0]]]
    scheme = quadpy.quadrilateral.Dunavant(4)
    assert scheme.degree >= 2*n
    vals = quadpy.quadrilateral.integrate(ff, quad, scheme)
    assert numpy.all(abs(vals - 1) < tol)
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
