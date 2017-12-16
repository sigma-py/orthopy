# -*- coding: utf-8 -*-
#
from __future__ import division

import numpy
import pytest
import sympy
from sympy import sqrt, pi

import orthopy


def test_integral0(n=4, tol=1.0e-13):
    polar = sympy.Symbol('theta', real=True)
    azimuthal = sympy.Symbol('phi', real=True)
    tree = numpy.concatenate(
            orthopy.sphere.sph_tree(
                n, polar, azimuthal, normalization='quantum mechanic',
                symbolic=True
                ))

    assert sympy.integrate(
        tree[0] * sympy.sin(polar), (polar, 0, pi), (azimuthal, 0, 2*pi)
        ) == 2*sqrt(pi)
    for val in tree[1:]:
        assert sympy.integrate(
            val * sympy.sin(polar), (azimuthal, 0, 2*pi), (polar, 0, pi)
            ) == 0
    return


def test_normality(n=3, tol=1.0e-13):
    '''Make sure that the polynomials are normal.
    '''
    polar = sympy.Symbol('theta', real=True)
    azimuthal = sympy.Symbol('phi', real=True)
    tree = numpy.concatenate(
            orthopy.sphere.sph_tree(
                n, polar, azimuthal, normalization='quantum mechanic',
                symbolic=True
                ))

    for val in tree:
        integrand = sympy.simplify(
            val * sympy.conjugate(val) * sympy.sin(polar)
            )
        assert sympy.integrate(
            integrand,
            (azimuthal, 0, 2*pi), (polar, 0, pi)
            ) == 1
    return


@pytest.mark.parametrize(
    'normalization', ['quantum mechanic', 'schmidt']
    )
def test_orthogonality(normalization, n=4, tol=1.0e-13):
    polar = sympy.Symbol('theta', real=True)
    azimuthal = sympy.Symbol('phi', real=True)
    tree = numpy.concatenate(
            orthopy.sphere.sph_tree(
                n, polar, azimuthal, normalization=normalization,
                symbolic=True
                ))
    vals = tree * sympy.conjugate(numpy.roll(tree, 1, axis=0))

    for val in vals:
        integrand = sympy.simplify(val * sympy.sin(polar))
        assert sympy.integrate(
            integrand,
            (azimuthal, 0, 2*pi), (polar, 0, pi)
            ) == 0
    return


def test_schmidt_normality(n=3, tol=1.0e-12):
    '''Make sure that the polynomials are orthonormal.
    '''
    polar = sympy.Symbol('theta', real=True)
    azimuthal = sympy.Symbol('phi', real=True)
    tree = numpy.concatenate(
            orthopy.sphere.sph_tree(
                n, polar, azimuthal, normalization='schmidt',
                symbolic=True
                ))
    # split into levels
    levels = [
        tree[0:1], tree[1:4], tree[4:9], tree[9:16], tree[16:25]
        ]

    for k, level in enumerate(levels):
        for val in level:
            integrand = sympy.simplify(
                val * sympy.conjugate(val) * sympy.sin(polar)
                )
            assert sympy.integrate(
                integrand,
                (azimuthal, 0, 2*pi), (polar, 0, pi)
                ) == 4*pi / (2*k+1)
    return


# pylint: disable=too-many-locals
def sph_exact2(theta, phi):
    # Exact values from
    # <https://en.wikipedia.org/wiki/Table_of_spherical_harmonics>.
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

    i = sympy.I

    # pylint: disable=invalid-unary-operand-type
    y1m1 = +sin_theta * exp(-i*phi) * sqrt(3 / pi / 2) / 2
    y1_0 = +cos_theta * sqrt(3 / pi) / 2
    y1p1 = -sin_theta * exp(+i*phi) * sqrt(3 / pi / 2) / 2
    #
    y2m2 = +sin_theta**2 * exp(-i*2*phi) * sqrt(15 / pi / 2) / 4
    y2m1 = +(sin_theta * cos_theta * exp(-i*phi)) * (sqrt(15 / pi / 2) / 2)
    y2_0 = +(3*cos_theta**2 - 1) * sqrt(5 / pi) / 4
    y2p1 = -(sin_theta * cos_theta * exp(+i*phi)) * (sqrt(15 / pi / 2) / 2)
    y2p2 = +sin_theta**2 * exp(+i*2*phi) * sqrt(15 / pi / 2) / 4
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
    vals = orthopy.sphere.sph_tree(
            L, theta, phi, normalization='quantum mechanic', symbolic=True
            )

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
    vals = orthopy.sphere.sph_tree(
            L, theta, phi, normalization='quantum mechanic'
            )

    cmplx = numpy.vectorize(complex)
    for val, ex in zip(vals, exacts):
        assert numpy.all(abs(val - cmplx(ex)) < 1.0e-12)
    return


def test_write():
    def sph22(polar, azimuthal):
        out = orthopy.sphere.sph_tree(
            5, polar, azimuthal, normalization='quantum mechanic'
            )[5][3]
        # out = numpy.arctan2(numpy.imag(out), numpy.real(out))
        out = abs(out)
        return out

    orthopy.sphere.write('sph.vtu', sph22)
    return


if __name__ == '__main__':
    test_schmidt_normality()
