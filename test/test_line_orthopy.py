# -*- coding: utf-8 -*-
#
from __future__ import division, print_function

import math

import numpy
import pytest
from scipy.special import legendre
import sympy

import orthopy


@pytest.mark.parametrize(
    'dtype', [numpy.float, sympy.S]
    )
def test_jacobi(dtype):
    n = 5
    if dtype == sympy.S:
        a = sympy.S(1)/1
        b = sympy.S(1)/1
        _, _, alpha, beta = \
            orthopy.line_segment.recurrence_coefficients.jacobi(
                n, a, b, 'monic'
                )
        assert all([a == 0 for a in alpha])
        assert (beta == [
            sympy.S(4)/3,
            sympy.S(1)/5,
            sympy.S(8)/35,
            sympy.S(5)/21,
            sympy.S(8)/33,
            ]).all()
    else:
        a = 1.0
        b = 1.0
        tol = 1.0e-14
        _, _, alpha, beta = \
            orthopy.line_segment.recurrence_coefficients.jacobi(
                n, a, b, 'monic'
                )
        assert numpy.all(abs(alpha) < tol)
        assert numpy.all(
            abs(beta - [4.0/3.0, 1.0/5.0, 8.0/35.0, 5.0/21.0, 8.0/33.0])
            < tol
            )
    return


@pytest.mark.parametrize(
    't, ref', [
        (sympy.S(1)/2, sympy.S(23)/2016),
        (1, sympy.S(8)/63),
        ]
    )
def test_eval(t, ref, tol=1.0e-14):
    n = 5
    p0, a, b, c = orthopy.line_segment.recurrence_coefficients.legendre(
            n, 'monic', symbolic=True
            )
    value = orthopy.line_segment.evaluate_orthogonal_polynomial(t, p0, a, b, c)

    assert value == ref

    # Evaluating the Legendre polynomial in this way is rather unstable, so
    # don't go too far with n.
    approx_ref = numpy.polyval(legendre(n, monic=True), t)
    assert abs(value - approx_ref) < tol
    return


@pytest.mark.parametrize(
    't, ref', [
        (
            numpy.array([1]),
            numpy.array([sympy.S(8)/63])
        ),
        (
            numpy.array([1, 2]),
            numpy.array([sympy.S(8)/63, sympy.S(1486)/63])
        ),
        ],
    )
def test_eval_vec(t, ref, tol=1.0e-14):
    n = 5
    p0, a, b, c = orthopy.line_segment.recurrence_coefficients.legendre(
            n, 'monic', symbolic=True
            )
    value = orthopy.line_segment.evaluate_orthogonal_polynomial(t, p0, a, b, c)

    assert (value == ref).all()

    # Evaluating the Legendre polynomial in this way is rather unstable, so
    # don't go too far with n.
    approx_ref = numpy.polyval(legendre(n, monic=True), t)
    assert (abs(value - approx_ref) < tol).all()
    return


def test_clenshaw(tol=1.0e-14):
    n = 5
    _, _, alpha, beta = \
        orthopy.line_segment.recurrence_coefficients.legendre(n, 'monic')
    t = 1.0

    a = numpy.ones(n+1)
    value = orthopy.line_segment.clenshaw(a, alpha, beta, t)

    ref = math.fsum([
            numpy.polyval(legendre(i, monic=True), t)
            for i in range(n+1)])

    assert abs(value - ref) < tol
    return


if __name__ == '__main__':
    # test_gauss('mpmath')
    # test_logo()
    pass
