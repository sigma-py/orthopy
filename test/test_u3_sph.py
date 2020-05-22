import itertools

import numpy
import pytest
import sympy
from sympy import pi, sqrt

import orthopy

polar = sympy.Symbol("theta", real=True)
azimuthal = sympy.Symbol("phi", real=True)


def _integrate(f):
    return sympy.integrate(
        sympy.simplify(f * sympy.sin(polar)), (azimuthal, 0, 2 * pi), (polar, 0, pi)
    )


def test_integral0(n=4):
    iterator = orthopy.u3.Iterator(
        polar, azimuthal, scaling="quantum mechanic", symbolic=True
    )
    out = next(iterator)
    assert _integrate(out[0]) == 2 * sqrt(pi)

    for _ in range(n):
        for val in next(iterator):
            assert _integrate(val) == 0


def test_normality(n=3):
    scaling = "quantum mechanic"
    iterator = itertools.islice(
        orthopy.u3.Iterator(polar, azimuthal, scaling, symbolic=True), n
    )
    for level in iterator:
        for val in level:
            assert _integrate(val * val.conjugate()) == 1


@pytest.mark.parametrize("scaling", ["quantum mechanic", "schmidt"])
def test_orthogonality(scaling, n=4):
    tree = numpy.concatenate(
        orthopy.u3.tree_sph(n, polar, azimuthal, scaling=scaling, symbolic=True)
    )
    vals = tree * sympy.conjugate(numpy.roll(tree, 1, axis=0))

    for val in vals:
        assert _integrate(val) == 0


def test_schmidt_seminormality(n=3):
    iterator = itertools.islice(
        orthopy.u3.Iterator(polar, azimuthal, scaling="schmidt", symbolic=True), n
    )
    for k, level in enumerate(iterator):
        for val in level:
            assert _integrate(val * val.conjugate()) == 4 * pi / (2 * k + 1)


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

    y1m1 = +sin_theta * exp(-i * phi) * sqrt(3 / pi / 2) / 2
    y1_0 = +cos_theta * sqrt(3 / pi) / 2
    y1p1 = -sin_theta * exp(+i * phi) * sqrt(3 / pi / 2) / 2
    #
    y2m2 = +(sin_theta ** 2) * exp(-i * 2 * phi) * sqrt(15 / pi / 2) / 4
    y2m1 = +(sin_theta * cos_theta * exp(-i * phi)) * (sqrt(15 / pi / 2) / 2)
    y2_0 = +(3 * cos_theta ** 2 - 1) * sqrt(5 / pi) / 4
    y2p1 = -(sin_theta * cos_theta * exp(+i * phi)) * (sqrt(15 / pi / 2) / 2)
    y2p2 = +(sin_theta ** 2) * exp(+i * 2 * phi) * sqrt(15 / pi / 2) / 4
    return [[y0_0], [y1m1, y1_0, y1p1], [y2m2, y2m1, y2_0, y2p1, y2p2]]


@pytest.mark.parametrize(
    "theta,phi",
    [
        (sympy.S(1) / 10, sympy.S(16) / 5),
        (sympy.S(1) / 10000, sympy.S(7) / 100000),
        # (
        #   numpy.array([sympy.S(3)/7, sympy.S(1)/13]),
        #   numpy.array([sympy.S(2)/5, sympy.S(2)/3]),
        # )
    ],
)
def test_spherical_harmonics(theta, phi):
    L = 2
    exacts = sph_exact2(theta, phi)
    vals = orthopy.u3.tree_sph(L, theta, phi, scaling="quantum mechanic", symbolic=True)

    for val, ex in zip(vals, exacts):
        for v, e in zip(val, ex):
            assert numpy.all(numpy.array(sympy.simplify(v - e)) == 0)


@pytest.mark.parametrize("theta,phi", [(1.0e-1, 16.0 / 5.0), (1.0e-4, 7.0e-5)])
def test_spherical_harmonics_numpy(theta, phi):
    L = 2
    exacts = sph_exact2(theta, phi)
    vals = orthopy.u3.tree_sph(L, theta, phi, scaling="quantum mechanic")

    cmplx = numpy.vectorize(complex)
    for val, ex in zip(vals, exacts):
        assert numpy.all(abs(val - cmplx(ex)) < 1.0e-12)


def test_write():
    def sph22(polar, azimuthal):
        return orthopy.u3.tree_sph(5, polar, azimuthal, scaling="quantum mechanic")[5][
            3
        ]

    orthopy.u3.write("sph.vtk", sph22)


if __name__ == "__main__":
    test_normality(n=3)
