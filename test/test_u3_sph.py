import itertools

import numpy
import pytest
import sympy
from sympy import pi, sqrt

import orthopy

polar = sympy.Symbol("theta", real=True)
azimuthal = sympy.Symbol("phi", real=True)


def _integrate(f):
    return sympy.integrate(f * sympy.sin(polar), (azimuthal, 0, 2 * pi), (polar, 0, pi))


def test_integral0(n=3):
    iterator = orthopy.u3.Eval(
        polar, azimuthal, scaling="quantum mechanic", symbolic=True
    )
    out = next(iterator)
    assert _integrate(out[0]) == 2 * sqrt(pi)

    for _ in range(n):
        for val in next(iterator):
            assert _integrate(val) == 0


def test_normality(n=3):
    scaling = "quantum mechanic"
    iterator = orthopy.u3.Eval(polar, azimuthal, scaling, symbolic=True)
    for level in itertools.islice(iterator, n):
        for val in level:
            assert _integrate(sympy.simplify(val * val.conjugate())) == 1


@pytest.mark.parametrize("scaling", ["quantum mechanic", "schmidt"])
def test_orthogonality(scaling, n=3):
    tree = numpy.concatenate(
        orthopy.u3.tree(n, polar, azimuthal, scaling=scaling, symbolic=True)
    )
    vals = tree * sympy.conjugate(numpy.roll(tree, 1, axis=0))

    for val in vals:
        assert _integrate(sympy.expand(val)) == 0


def test_schmidt_seminormality(n=3):
    iterator = orthopy.u3.Eval(polar, azimuthal, scaling="schmidt", symbolic=True)
    for k, level in enumerate(itertools.islice(iterator, n)):
        ref = 4 * pi / (2 * k + 1)
        for val in level:
            assert _integrate(sympy.simplify(val * val.conjugate())) == ref


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
    vals = orthopy.u3.tree(L, theta, phi, scaling="quantum mechanic", symbolic=True)

    for val, ex in zip(vals, exacts):
        for v, e in zip(val, ex):
            assert numpy.all(numpy.array(sympy.simplify(v - e)) == 0)


@pytest.mark.parametrize("theta,phi", [(1.0e-1, 16.0 / 5.0), (1.0e-4, 7.0e-5)])
def test_spherical_harmonics_numpy(theta, phi):
    L = 2
    exacts = sph_exact2(theta, phi)
    vals = orthopy.u3.tree(L, theta, phi, scaling="quantum mechanic")

    cmplx = numpy.vectorize(complex)
    for val, ex in zip(vals, exacts):
        assert numpy.all(abs(val - cmplx(ex)) < 1.0e-12)


def test_write():
    def sph22(polar, azimuthal):
        return orthopy.u3.tree(5, polar, azimuthal, scaling="quantum mechanic")[5][3]

    orthopy.u3.write("sph.vtk", sph22)


# from associated_legendre
# @pytest.mark.parametrize(
#     "x",
#     [
#         sympy.S(1) / 10,
#         sympy.S(1) / 1000,
#         numpy.array([sympy.S(3) / 7, sympy.S(1) / 13]),
#     ],
# )
# @pytest.mark.parametrize(
#     "scaling,factor",
#     [
#         (
#             "spherical",
#             # sqrt((2*L+1) / 4 / pi * factorial(l-m) / factorial(l+m))
#             lambda L, m: sympy.sqrt(sympy.S(2 * L + 1) / (4 * sympy.pi) * ff(L, m)),
#         ),
#         ("schmidt", lambda L, m: 2 * sympy.sqrt(ff(L, m))),
#     ],
# )
# def test_exact(x, scaling, factor):
#     """Test for the exact values.
#     """
#     L = 4
#     vals = orthopy.c1.associated_legendre.tree(L, x, scaling, symbolic=True)
#
#     exacts = exact_natural(x)
#     exacts = [
#         [val * factor(L, m - L) for m, val in enumerate(ex)]
#         for L, ex in enumerate(exacts)
#     ]
#
#     for val, ex in zip(vals, exacts):
#         for v, e in zip(val, ex):
#             assert numpy.all(v == e)


if __name__ == "__main__":
    test_normality(n=3)
