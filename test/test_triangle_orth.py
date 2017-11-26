# -*- coding: utf-8 -*-
#
from __future__ import division

import numpy
import orthopy
import quadpy
import pytest
import scipy.special


def op(i, j, x, y):
    p0, a, b, c = orthopy.line.recurrence_coefficients.jacobi(
            i, 0, 0,
            # standardization='monic'
            standardization='p(1)=(n+alpha over n)'
            )
    val1 = orthopy.line.tools.evaluate_orthogonal_polynomial(
            (x-y)/(x+y), p0, a, b, c
            )

    val1 = numpy.polyval(scipy.special.jacobi(i, 0, 0), (x-y)/(x+y))

    # treat x==0, y==0 separately
    if isinstance(val1, numpy.ndarray):
        idx = numpy.where(numpy.logical_and(x == 0, y == 0))[0]
        val1[idx] = numpy.polyval(scipy.special.jacobi(i, 0, 0), 0.0)
    else:
        if numpy.isnan(val1):
            val1 = numpy.polyval(scipy.special.jacobi(i, 0, 0), 0.0)

    p0, a, b, c = orthopy.line.recurrence_coefficients.jacobi(
            j, 2*i+1, 0,
            # standardization='monic'
            standardization='p(1)=(n+alpha over n)'
            )
    val2 = orthopy.line.tools.evaluate_orthogonal_polynomial(
            1-2*(x+y), p0, a, b, c
            )
    # val2 = numpy.polyval(scipy.special.jacobi(j, 2*i+1, 0), 1-2*(x+y))

    flt = numpy.vectorize(float)
    return flt(
        numpy.sqrt(2*i + 1) * val1 * (x+y)**i
        * numpy.sqrt(2*j + 2*i + 2) * val2
        )


def eval_orthpolys4(bary):
    '''Evaluate all orthogonal polynomials at x.
    See, e.g.,

    S.-A. Papanicolopulos,
    New fully symmetric and rotationally symmetric cubature rules on the
    triangle using minimal orthonormal bases,
    <https://arxiv.org/pdf/1411.5631.pdf>.
    '''
    x, y = bary[0], bary[1]

    def f(i, j):
        return op(i, j, x, y)

    # pylint: disable=invalid-unary-operand-type
    return [
        [+f(0, 0)],
        [-f(0, 1), +f(1, 0)],
        [+f(0, 2), -f(1, 1), +f(2, 0)],
        [-f(0, 3), +f(1, 2), -f(2, 1), +f(3, 0)],
        [+f(0, 4), -f(1, 3), +f(2, 2), -f(3, 1), +f(4, 0)],
        ]


@pytest.mark.parametrize(
    'x', [
        numpy.array([0.24, 0.65]),
        numpy.random.rand(2, 5),
        ])
def test_triangle_orth(x, tol=1.0e-12):
    L = 4
    exacts = eval_orthpolys4(x)

    # print('exact:')
    # for ex in exacts:
    #     print(ex)
    # print

    bary = numpy.array([
        x[0], x[1], 1.0-x[0]-x[1]
        ])
    vals = orthopy.triangle.orth_tree(L, bary, 'normal')

    # print('tree:')
    # for ex in vals:
    #     print(ex)
    # print

    for val, ex in zip(vals, exacts):
        for v, e in zip(val, ex):
            assert numpy.all(numpy.abs(v - e) < tol * numpy.abs(e))
    return


def test_orthonormal(tol=1.0e-14):
    '''Make sure that the polynomials are orthonormal
    '''
    n = 4
    # Choose a scheme of order 2*n.
    scheme = quadpy.triangle.Dunavant(8)

    triangle = numpy.array([[0, 0], [1, 0], [0, 1]])

    # normality
    def ff(x):
        bary = numpy.array([x[0], x[1], 1-x[0]-x[1]])
        tree = numpy.concatenate(orthopy.triangle.orth_tree(n, bary, 'normal'))
        return tree * tree

    val = quadpy.triangle.integrate(ff, triangle, scheme)
    assert numpy.all(abs(val - 1) < tol)

    # orthogonality
    def f_shift(x):
        bary = numpy.array([x[0], x[1], 1-x[0]-x[1]])
        tree = numpy.concatenate(orthopy.triangle.orth_tree(n, bary, 'normal'))
        return tree * numpy.roll(tree, 1, axis=0)

    val = quadpy.triangle.integrate(f_shift, triangle, scheme)
    assert numpy.all(abs(val) < tol)
    return


def test_show(n=2, r=1):
    # plot the triangle
    alpha = numpy.pi * numpy.array([3.0/6.0, 7.0/6.0, 11.0/6.0])
    corners = numpy.array([numpy.cos(alpha), numpy.sin(alpha)])

    # corners = numpy.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]).T

    def f(bary):
        return orthopy.triangle.orth_tree(n, bary, 'normal')[n][r]

    orthopy.triangle.show(corners, f)
    # orthopy.triangle.plot(corners, f)
    # import matplotlib.pyplot as plt
    # plt.savefig('triangle.png', transparent=True)
    return


if __name__ == '__main__':
    # x_ = numpy.array([0.24, 0.65])
    # # x_ = numpy.random.rand(3, 2)
    # test_triangle_orth(x=x_)
    test_show()
    # test_orthonormal()
