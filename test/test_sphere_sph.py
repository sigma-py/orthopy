# -*- coding: utf-8 -*-
#
from __future__ import division

import numpy
from numpy import sqrt, pi, exp
import orthopy
import pytest


def sph_exact2(theta, phi):
    try:
        assert numpy.all(theta.shape == phi.shape)
        y0_0 = numpy.full(theta.shape, 0.5 * sqrt(1 / pi))
    except AttributeError:
        y0_0 = 0.5 * sqrt(1 / pi)

    sin_theta = numpy.sin(theta)
    cos_theta = numpy.cos(theta)

    y1m1 = +0.5 * sqrt(3 / 2 / pi) * sin_theta * exp(-1j*phi)
    y1_0 = 0.5 * sqrt(3 / pi) * cos_theta
    y1p1 = -0.5 * sqrt(3 / 2 / pi) * sin_theta * exp(+1j*phi)
    #
    y2m2 = +0.25 * sqrt(15 / 2 / pi) * sin_theta**2 * exp(-1j*2*phi)
    y2m1 = +0.50 * sqrt(15 / 2 / pi) * sin_theta * cos_theta * exp(-1j*phi)
    y2_0 = +0.25 * sqrt(5 / pi) * (3*cos_theta**2 - 1)
    y2p1 = -0.50 * sqrt(15 / 2 / pi) * sin_theta * cos_theta * exp(+1j*phi)
    y2p2 = +0.25 * sqrt(15 / 2 / pi) * sin_theta**2 * exp(+1j*2*phi)
    return [
        [y0_0],
        [y1m1, y1_0, y1p1],
        [y2m2, y2m1, y2_0, y2p1, y2p2],
        ]


@pytest.mark.parametrize(
    'theta,phi', [
        (1.0e-1, 3.2e-1),
        (1.0e-4, 0.7e-4),
        (numpy.random.rand(3, 2), numpy.random.rand(3, 2)),
        ]
    )
def test_spherical_harmonics(theta, phi, tol=1.0e-8):
    L = 2
    vals = orthopy.sphere.sph_tree(L, theta, phi)
    exacts = sph_exact2(theta, phi)

    for val, ex in zip(vals, exacts):
        for v, e in zip(val, ex):
            assert numpy.all(numpy.abs(v - e) < tol * numpy.abs(e))
    return


if __name__ == '__main__':
    test_spherical_harmonics(
        # 1.0e-1, 3.2e-1
        1.0e-4, 0.7e-4
        )
