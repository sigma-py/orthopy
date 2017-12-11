# -*- coding: utf-8 -*-
#
from __future__ import division

import numpy
import orthopy
import pytest
import sympy
from sympy import sqrt, pi


# pylint: disable=too-many-locals
def sph_exact2(theta, phi):
    try:
        assert numpy.all(theta.shape == phi.shape)
        y0_0 = numpy.full(theta.shape, sqrt(1 / pi) / 2)
    except AttributeError:
        y0_0 = sqrt(1 / pi) / 2

    sin = numpy.vectorize(sympy.sin)
    cos = numpy.vectorize(sympy.cos)
    exp = numpy.vectorize(sympy.exp)

    sin_theta = sin(theta)
    cos_theta = cos(theta)

    # pylint: disable=invalid-unary-operand-type
    y1m1 = +sin_theta * exp(-1j*phi) * sqrt(3 / pi / 2) / 2
    y1_0 = +cos_theta * sqrt(3 / pi) / 2
    y1p1 = -sin_theta * exp(+1j*phi) * sqrt(3 / pi / 2) / 2
    #
    y2m2 = +sin_theta**2 * exp(-1j*2*phi) * sqrt(15 / pi / 2) / 4
    y2m1 = +(sin_theta * cos_theta * exp(-1j*phi)) * (sqrt(15 / pi / 2) / 2)
    y2_0 = +(3*cos_theta**2 - 1) * sqrt(5 / pi) / 4
    y2p1 = -(sin_theta * cos_theta * exp(+1j*phi)) * (sqrt(15 / pi / 2) / 2)
    y2p2 = +sin_theta**2 * exp(+1j*2*phi) * sqrt(15 / pi / 2) / 4
    return [
        [y0_0],
        [y1m1, y1_0, y1p1],
        [y2m2, y2m1, y2_0, y2p1, y2p2],
        ]


@pytest.mark.parametrize(
    'theta,phi', [
        (sympy.Rational(1, 10), sympy.Rational(16, 5)),
        (sympy.Rational(1, 10000), sympy.Rational(7, 100000)),
        # (
        #   numpy.array([sympy.Rational(3, 7), sympy.Rational(1, 13)]),
        #   numpy.array([sympy.Rational(2, 5), sympy.Rational(2, 3)]),
        # )
        ]
    )
def test_spherical_harmonics(theta, phi):
    L = 2
    exacts = sph_exact2(theta, phi)
    vals = orthopy.sphere.sph_tree(L, theta, phi, symbolic=True)

    for val, ex in zip(vals, exacts):
        for v, e in zip(val, ex):
            assert numpy.all(numpy.array(sympy.simplify(v - e)) == 0)
    return


@pytest.mark.parametrize(
    'theta,phi', [
        (1.0e-1, 16.0/5.0),
        (1.0e-4, 7.0e-5),
        ]
    )
def test_spherical_harmonics_numpy(theta, phi):
    L = 2
    exacts = sph_exact2(theta, phi)
    vals = orthopy.sphere.sph_tree(L, theta, phi)

    cmplx = numpy.vectorize(complex)
    for val, ex in zip(vals, exacts):
        assert numpy.all(abs(val - cmplx(ex)) < 1.0e-12)
    return


if __name__ == '__main__':
    test_spherical_harmonics(
        # 1.0e-1, 3.2e-1
        1.0e-4, 0.7e-4
        )
