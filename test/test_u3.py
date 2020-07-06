import itertools

import numpy
import pytest
import sympy
from sympy import pi, sqrt

import ndim
import orthopy

X = sympy.symbols("x, y, z")
P = [sympy.poly(x, X) for x in X]
polar = sympy.Symbol("theta", real=True)
azimuthal = sympy.Symbol("phi", real=True)


# def _integrate(f):
#     return sympy.integrate(f * sympy.sin(polar), (azimuthal, 0, 2 * pi), (polar, 0, pi))


def _integrate_poly(p):
    return sum(
        c * ndim.nsphere.integrate_monomial(k, symbolic=True)
        for c, k in zip(p.coeffs(), p.monoms())
    )


@pytest.mark.parametrize(
    "scaling,int0",
    [
        ("acoustic", 2 * sqrt(pi)),
        ("geodetic", 4 * pi),
        ("quantum mechanic", 2 * sqrt(pi)),
        ("schmidt", 4 * pi),
    ],
)
def test_integral0(scaling, int0, n=5):
    iterator = orthopy.u3.Eval(P, scaling, symbolic=True)
    for k, vals in enumerate(itertools.islice(iterator, n)):
        for val in vals:
            assert _integrate_poly(val) == (int0 if k == 0 else 0)


def _conj(p):
    # https://stackoverflow.com/a/62325596/353337
    # https://github.com/sympy/sympy/issues/19531
    return sympy.Poly.from_dict(
        {m: p.coeff_monomial(m).conjugate() for m in p.monoms()}, p.gens
    )


@pytest.mark.parametrize("scaling", ["acoustic", "quantum mechanic"])
def test_normality(scaling, n=5):
    # iterator = orthopy.u3.EvalPolar(polar, azimuthal, scaling, symbolic=True)
    iterator = orthopy.u3.Eval(P, scaling, symbolic=True)
    for k, level in enumerate(itertools.islice(iterator, n)):
        for val in level:
            assert _integrate_poly(val * _conj(val)) == 1


@pytest.mark.parametrize(
    "scaling", ["acoustic", "geodetic", "quantum mechanic", "schmidt"]
)
def test_orthogonality(scaling, n=3):
    tree = numpy.concatenate(orthopy.u3.tree(n, P, scaling=scaling, symbolic=True))
    for f0, f1 in itertools.combinations(tree, 2):
        assert _integrate_poly(f0 * _conj(f1)) == 0


def test_schmidt_seminormality(n=3):
    iterator = orthopy.u3.Eval(P, scaling="schmidt", symbolic=True)
    for k, level in enumerate(itertools.islice(iterator, n)):
        ref = 4 * pi / (2 * k + 1)
        for val in level:
            assert _integrate_poly(val * _conj(val)) == ref


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
    vals = orthopy.u3.tree_polar(
        L, theta, phi, scaling="quantum mechanic", symbolic=True
    )

    for val, ex in zip(vals, exacts):
        for v, e in zip(val, ex):
            assert numpy.all(numpy.array(sympy.simplify(v - e)) == 0)


@pytest.mark.parametrize("theta,phi", [(1.0e-1, 16.0 / 5.0), (1.0e-4, 7.0e-5)])
def test_spherical_harmonics_numpy(theta, phi):
    L = 2
    exacts = sph_exact2(theta, phi)
    vals = orthopy.u3.tree_polar(L, theta, phi, scaling="quantum mechanic")

    cmplx = numpy.vectorize(complex)
    for val, ex in zip(vals, exacts):
        assert numpy.all(abs(val - cmplx(ex)) < 1.0e-12)


def test_write(n=5, r=3):
    orthopy.u3.write_single(f"sph{n}{r}.vtk", n, r, "quantum mechanic")


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
    test_write()
