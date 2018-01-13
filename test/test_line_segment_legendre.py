# -*- coding: utf-8 -*-
#
import numpy
import pytest
import scipy.special
import sympy
from sympy import S, sqrt

import orthopy


@pytest.mark.parametrize('n, y', [
    (0, [1, 1, 1]),
    (1, [0, S(1)/2, 1]),
    (2, [-S(1)/3, -S(1)/12, S(2)/3]),
    (3, [0, -S(7)/40, S(2)/5]),
    (4, [S(3)/35, -S(37)/560, S(8)/35]),
    (5, [0, S(23)/2016, S(8)/63]),
    ]
    )
def test_legendre_monic(n, y):
    x = numpy.array([0, S(1)/2, 1])

    out = orthopy.line_segment.recurrence_coefficients.legendre(
            n, 'monic', symbolic=True
            )

    # Test evaluation of one value
    y0 = orthopy.tools.line_evaluate(x[0], *out)
    assert y0 == y[0]

    # Test evaluation of multiple values
    val = orthopy.tools.line_evaluate(x, *out)
    assert all(val == y)
    return


@pytest.mark.parametrize('n, y', [
    (0, [1, 1, 1]),
    (1, [0, S(1)/2, 1]),
    (2, [-S(1)/2, -S(1)/8, 1]),
    (3, [0, -S(7)/16, 1]),
    (4, [S(3)/8, -S(37)/128, 1]),
    (5, [0, S(23)/256, 1]),
    ]
    )
def test_legendre_p11(n, y):
    x = numpy.array([0, S(1)/2, 1])

    out = orthopy.line_segment.recurrence_coefficients.legendre(
            n, standardization='p(1)=1'
            )

    y0 = orthopy.tools.line_evaluate(x[0], *out)
    assert y0 == y[0]

    val = orthopy.tools.line_evaluate(x, *out)
    assert all(val == y)
    return


@pytest.mark.parametrize('n, y', [
    (0, [sqrt(S(1)/2), sqrt(S(1)/2), sqrt(S(1)/2)]),
    (1, [0, sqrt(S(3)/8), sqrt(S(3)/2)]),
    (2, [-sqrt(S(5)/8), -sqrt(S(5)/128), sqrt(S(5)/2)]),
    (3, [0, -sqrt(S(343)/512), sqrt(S(7)/2)]),
    (4, [9 / sqrt(2) / 8, -111 / sqrt(2) / 128, 3/sqrt(2)]),
    (5, [0, sqrt(S(5819)/131072), sqrt(S(11)/2)]),
    ]
    )
def test_legendre_normal(n, y):
    x = numpy.array([0, S(1)/2, 1])

    out = orthopy.line_segment.recurrence_coefficients.legendre(
            n, standardization='normal', symbolic=True
            )

    y0 = orthopy.tools.line_evaluate(x[0], *out)
    assert y0 == y[0]

    val = orthopy.tools.line_evaluate(x, *out)
    assert all(val == y)
    return


def test_integral0(n=4):
    x = sympy.Symbol('x')
    vals = orthopy.line_segment.tree_legendre(n, 'normal', x, symbolic=True)

    assert sympy.integrate(vals[0], (x, -1, +1)) == sqrt(2)
    for val in vals[1:]:
        assert sympy.integrate(val, (x, -1, +1)) == 0
    return


def test_normality(n=4):
    x = sympy.Symbol('x')
    vals = orthopy.line_segment.tree_legendre(n, 'normal', x, symbolic=True)

    for val in vals:
        assert sympy.integrate(val**2, (x, -1, +1)) == 1
    return


def test_orthogonality(n=4):
    x = sympy.Symbol('x')
    vals = orthopy.line_segment.tree_legendre(n, 'normal', x, symbolic=True)
    out = vals * numpy.roll(vals, 1, axis=0)

    for val in out:
        assert sympy.integrate(val, (x, -1, +1)) == 0
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
    value = orthopy.tools.line_evaluate(t, p0, a, b, c)

    assert value == ref

    # Evaluating the Legendre polynomial in this way is rather unstable, so
    # don't go too far with n.
    approx_ref = numpy.polyval(scipy.special.legendre(n, monic=True), t)
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
    value = orthopy.tools.line_evaluate(t, p0, a, b, c)

    assert (value == ref).all()

    # Evaluating the Legendre polynomial in this way is rather unstable, so
    # don't go too far with n.
    approx_ref = numpy.polyval(scipy.special.legendre(n, monic=True), t)
    assert (abs(value - approx_ref) < tol).all()
    return


if __name__ == '__main__':
    test_normality()
