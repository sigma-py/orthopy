# -*- coding: utf-8 -*-
#
from __future__ import division

import numpy
import pytest
import scipy.special
import sympy
from sympy import S

import orthopy


def op(i, j, x, y):
    p0, a, b, c = orthopy.line_segment.recurrence_coefficients.jacobi(
            i, 0, 0,
            # standardization='monic'
            standardization='p(1)=(n+alpha over n)'
            )
    val1 = orthopy.tools.line_evaluate((x-y)/(x+y), p0, a, b, c)

    val1 = numpy.polyval(scipy.special.jacobi(i, 0, 0), (x-y)/(x+y))

    # treat x==0, y==0 separately
    if isinstance(val1, numpy.ndarray):
        idx = numpy.where(numpy.logical_and(x == 0, y == 0))[0]
        val1[idx] = numpy.polyval(scipy.special.jacobi(i, 0, 0), 0.0)
    else:
        if numpy.isnan(val1):
            val1 = numpy.polyval(scipy.special.jacobi(i, 0, 0), 0.0)

    p0, a, b, c = orthopy.line_segment.recurrence_coefficients.jacobi(
            j, 2*i+1, 0,
            # standardization='monic'
            standardization='p(1)=(n+alpha over n)'
            )
    val2 = orthopy.tools.line_evaluate(1-2*(x+y), p0, a, b, c)
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

    bary = numpy.array([
        x[0], x[1], 1-x[0]-x[1]
        ])
    vals = orthopy.triangle.tree(bary, L, 'normal')

    for val, ex in zip(vals, exacts):
        for v, e in zip(val, ex):
            assert numpy.all(numpy.abs(v - e) < tol * numpy.abs(e))
    return


def test_triangle_orth_exact():
    x = numpy.array([S(1)/3, S(1)/7])

    L = 2
    exacts = [
        [sympy.sqrt(2)],
        [-S(8)/7, 8*sympy.sqrt(3)/21],
        [-197*sympy.sqrt(6)/441,
         -136*sympy.sqrt(2)/147,
         -26*sympy.sqrt(30)/441],
        ]

    bary = numpy.array([
        x[0], x[1], 1-x[0]-x[1]
        ])
    vals = orthopy.triangle.tree(bary, L, 'normal', symbolic=True)

    for val, ex in zip(vals, exacts):
        for v, e in zip(val, ex):
            assert v == e
    return


def test_triangle_orth_1_exact():
    x = numpy.array([
        [S(1)/5, S(2)/5, S(3)/5],
        [S(1)/7, S(2)/7, S(3)/7],
        ])

    L = 2
    exacts = [
        [[1, 1, 1]],
        [
            [-S(34)/35, S(2)/35, S(38)/35],
            [S(2)/35, S(4)/35, S(6)/35]
        ],
        ]

    bary = numpy.array([
        x[0], x[1], 1-x[0]-x[1]
        ])
    vals = orthopy.triangle.tree(bary, L, '1', symbolic=True)

    for val, ex in zip(vals, exacts):
        for v, e in zip(val, ex):
            assert numpy.all(v == e)
    return


def test_integral0(n=4):
    b0 = sympy.Symbol('b0')
    b1 = sympy.Symbol('b1')
    b = numpy.array([b0, b1, 1-b0-b1])
    tree = numpy.concatenate(
        orthopy.triangle.tree(b, n, 'normal', symbolic=True)
        )

    assert \
        sympy.integrate(tree[0], (b0, 0, 1-b1), (b1, 0, 1)) == sympy.sqrt(2)/2
    for val in tree[1:]:
        assert sympy.integrate(val, (b0, 0, 1-b1), (b1, 0, 1)) == 0
    return


def test_normality(n=4):
    '''Make sure that the polynomials are orthonormal
    '''
    b0 = sympy.Symbol('b0')
    b1 = sympy.Symbol('b1')
    b = numpy.array([b0, b1, 1-b0-b1])
    tree = numpy.concatenate(
        orthopy.triangle.tree(b, n, 'normal', symbolic=True)
        )

    for val in tree:
        assert sympy.integrate(val**2, (b0, 0, 1-b1), (b1, 0, 1)) == 1
    return


def test_orthogonality(n=4):
    b0 = sympy.Symbol('b0')
    b1 = sympy.Symbol('b1')
    b = numpy.array([b0, b1, 1-b0-b1])
    tree = numpy.concatenate(
        orthopy.triangle.tree(b, n, 'normal', symbolic=True)
        )

    shifts = tree * numpy.roll(tree, 1, axis=0)

    for val in shifts:
        assert sympy.integrate(val, (b0, 0, 1-b1), (b1, 0, 1)) == 0
    return


def test_show(n=2, r=1):
    # plot the triangle
    alpha = numpy.pi * numpy.array([3.0/6.0, 7.0/6.0, 11.0/6.0])
    corners = numpy.array([numpy.cos(alpha), numpy.sin(alpha)])

    # corners = numpy.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]).T

    def f(bary):
        return orthopy.triangle.tree(bary, n, 'normal')[n][r]

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
