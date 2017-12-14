# -*- coding: utf-8 -*-
#
from __future__ import division

import numpy
import orthopy
import quadpy


def test_orthonormal(tol=1.0e-14):
    '''Make sure that the polynomials are orthonormal
    '''
    n = 4
    # Choose a scheme of order at least 2*n.
    scheme = quadpy.quadrilateral.Dunavant(4)

    quad = [[[-1.0, -1.0], [+1.0, -1.0]], [[-1.0, +1.0], [+1.0, +1.0]]]

    # normality
    def ff(x):
        tree = numpy.concatenate(
                orthopy.quadrilateral.orth_tree(n, x)
                )
        return tree * tree

    val = quadpy.quadrilateral.integrate(ff, quad, scheme)
    assert numpy.all(abs(val - 1) < tol)

    # orthogonality
    def f_shift(x):
        tree = numpy.concatenate(
                orthopy.triangle.orth_tree(n, x)
                )
        return tree * numpy.roll(tree, 1, axis=0)

    val = quadpy.quadrilateral.integrate(f_shift, quad, scheme)
    assert numpy.all(abs(val) < tol)
    return


# def test_show(n=2, r=1):
#     # plot the triangle
#     alpha = numpy.pi * numpy.array([3.0/6.0, 7.0/6.0, 11.0/6.0])
#     corners = numpy.array([numpy.cos(alpha), numpy.sin(alpha)])
#
#     # corners = numpy.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]).T
#
#     def f(bary):
#         return orthopy.triangle.orth_tree(n, bary, 'normal')[n][r]
#
#     orthopy.triangle.show(corners, f)
#     # orthopy.triangle.plot(corners, f)
#     # import matplotlib.pyplot as plt
#     # plt.savefig('triangle.png', transparent=True)
#     return


# if __name__ == '__main__':
#     # x_ = numpy.array([0.24, 0.65])
#     # # x_ = numpy.random.rand(3, 2)
#     # test_triangle_orth(x=x_)
#     test_show()
#     # test_orthonormal()
