# -*- coding: utf-8 -*-
#
from __future__ import division, print_function

from distutils.version import LooseVersion

import math

from mpmath import mp
import numpy
import pytest
import scipy
from scipy.special import legendre
import sympy

import orthopy


def test_golub_welsch(tol=1.0e-14):
    '''Test the custom Gauss generator with the weight function x**2.
    '''
    alpha = 2.0

    # Get the moment corresponding to the weight function omega(x) =
    # x^alpha:
    #
    #                                     / 0 if k is odd,
    #    int_{-1}^{+1} |x^alpha| x^k dx ={
    #                                     \ 2/(alpha+k+1) if k is even.
    #
    n = 5
    k = numpy.arange(2*n+1)
    moments = (1.0 + (-1.0)**k) / (k + alpha + 1)
    alpha, beta = orthopy.line.golub_welsch(moments)

    assert numpy.all(abs(alpha) < tol)
    assert abs(beta[0] - 2.0/3.0) < tol
    assert abs(beta[1] - 3.0/5.0) < tol
    assert abs(beta[2] - 4.0/35.0) < tol
    assert abs(beta[3] - 25.0/63.0) < tol
    assert abs(beta[4] - 16.0/99.0) < tol

    orthopy.line.check_coefficients(moments, alpha, beta)
    return


@pytest.mark.parametrize(
    'dtype', [numpy.float, sympy.S]
    )
def test_chebyshev(dtype):
    alpha = 2

    # Get the moment corresponding to the weight function omega(x) =
    # x^alpha:
    #
    #                                     / 0 if k is odd,
    #    int_{-1}^{+1} |x^alpha| x^k dx ={
    #                                     \ 2/(alpha+k+1) if k is even.
    #
    n = 5

    if dtype == sympy.S:
        moments = [
            sympy.S(1 + (-1)**kk) / (kk + alpha + 1)
            for kk in range(2*n)
            ]

        alpha, beta = orthopy.line.chebyshev(moments)

        assert all([a == 0 for a in alpha])
        assert (beta == [
            sympy.S(2)/3,
            sympy.S(3)/5,
            sympy.S(4)/35,
            sympy.S(25)/63,
            sympy.S(16)/99,
            ]).all()
    else:
        assert dtype == numpy.float
        tol = 1.0e-14
        k = numpy.arange(2*n)
        moments = (1.0 + (-1.0)**k) / (k + alpha + 1)

        alpha, beta = orthopy.line.chebyshev(moments)

        assert numpy.all(abs(alpha) < tol)
        assert numpy.all(
            abs(beta - [2.0/3.0, 3.0/5.0, 4.0/35.0, 25.0/63.0, 16.0/99.0])
            < tol
            )
    return


def test_chebyshev_modified(tol=1.0e-14):
    alpha = 2.0

    # Get the moments corresponding to the Legendre polynomials and the weight
    # function omega(x) = |x^alpha|:
    #
    #                                        / 2/3   if k == 0,
    #    int_{-1}^{+1} |x^alpha| P_k(x) dx ={  8/45  if k == 2,
    #                                        \ 0     otherwise.
    #
    n = 5
    moments = numpy.zeros(2*n)
    moments[0] = 2.0/3.0
    moments[2] = 8.0/45.0
    _, _, b, c = \
        orthopy.line.recurrence_coefficients.legendre(2*n, 'monic')

    alpha, beta = orthopy.line.chebyshev_modified(moments, b, c)

    assert numpy.all(abs(alpha) < tol)
    assert numpy.all(
        abs(beta - [2.0/3.0, 3.0/5.0, 4.0/35.0, 25.0/63.0, 16.0/99.0])
        < tol
        )
    return


@pytest.mark.parametrize(
    'dtype', [numpy.float, sympy.S]
    )
def test_jacobi(dtype):
    n = 5
    if dtype == sympy.S:
        a = sympy.S(1)/1
        b = sympy.S(1)/1
        _, _, alpha, beta = \
            orthopy.line.recurrence_coefficients.jacobi(
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
            orthopy.line.recurrence_coefficients.jacobi(
                n, a, b, 'monic'
                )
        assert numpy.all(abs(alpha) < tol)
        assert numpy.all(
            abs(beta - [4.0/3.0, 1.0/5.0, 8.0/35.0, 5.0/21.0, 8.0/33.0])
            < tol
            )
    return


@pytest.mark.parametrize(
    'mode', ['sympy', 'numpy', 'mpmath']
    )
def test_gauss(mode):
    if mode == 'sympy':
        n = 3
        a = sympy.S(0)/1
        b = sympy.S(0)/1
        _, _, alpha, beta = \
            orthopy.line.recurrence_coefficients.jacobi(
                n, a, b, 'monic', symbolic=True
                )
        points, weights = \
            orthopy.line.schemes.custom(alpha, beta, mode=mode)

        assert points == [
            -sympy.sqrt(sympy.S(3)/5),
            0,
            +sympy.sqrt(sympy.S(3)/5),
            ]

        assert weights == [
            sympy.S(5)/9,
            sympy.S(8)/9,
            sympy.S(5)/9,
            ]

    elif mode == 'mpmath':
        n = 5
        a = sympy.S(0)/1
        b = sympy.S(0)/1
        _, _, alpha, beta = \
            orthopy.line.recurrence_coefficients.jacobi(
                n, a, b, 'monic'
                )
        points, weights = orthopy.line.schemes.custom(
                alpha, beta,
                mode=mode,
                decimal_places=50
                )

        tol = 1.0e-50
        mp.dps = 50
        s = mp.sqrt(5 + 2*mp.sqrt(mp.mpf(10)/mp.mpf(7))) / 3
        t = mp.sqrt(5 - 2*mp.sqrt(mp.mpf(10)/mp.mpf(7))) / 3
        assert (abs(points - [-s, -t, 0.0, +t, +s]) < tol).all()

        u = mp.mpf(128) / mp.mpf(225)
        v = (322 + 13 * mp.sqrt(70)) / 900
        w = (322 - 13 * mp.sqrt(70)) / 900
        assert (abs(weights - [w, v, u, v, w]) < tol).all()

    else:
        assert mode == 'numpy'
        n = 5
        tol = 1.0e-14
        _, _, alpha, beta = \
            orthopy.line.recurrence_coefficients.legendre(n, 'monic')
        alpha = numpy.array([float(a) for a in alpha])
        beta = numpy.array([float(b) for b in beta])
        points, weights = orthopy.line.schemes.custom(
                alpha, beta,
                mode=mode
                )

        s = math.sqrt(5.0 + 2*math.sqrt(10.0/7.0)) / 3.0
        t = math.sqrt(5.0 - 2*math.sqrt(10.0/7.0)) / 3.0
        assert (abs(points - [-s, -t, 0.0, +t, +s]) < tol).all()

        u = 128.0/225.0
        v = (322.0 + 13 * math.sqrt(70)) / 900.0
        w = (322.0 - 13 * math.sqrt(70)) / 900.0
        assert (abs(weights - [w, v, u, v, w]) < tol).all()
    return


@pytest.mark.skipif(
    LooseVersion(scipy.__version__) < LooseVersion('1.0.0'),
    reason='Requires SciPy 1.0'
    )
def test_jacobi_reconstruction(tol=1.0e-14):
    _, _, alpha1, beta1 = \
        orthopy.line.recurrence_coefficients.jacobi(4, 2, 1, 'monic')
    points, weights = orthopy.line.schemes.custom(alpha1, beta1)

    alpha2, beta2 = \
        orthopy.line.coefficients_from_gauss(points, weights)

    assert numpy.all(abs(alpha1 - alpha2) < tol)
    assert numpy.all(abs(beta1 - beta2) < tol)
    return


@pytest.mark.parametrize(
    't, ref', [
        (sympy.S(1)/2, sympy.S(23)/2016),
        (1, sympy.S(8)/63),
        ]
    )
def test_eval(t, ref, tol=1.0e-14):
    n = 5
    p0, a, b, c = orthopy.line.recurrence_coefficients.legendre(
            n, 'monic', symbolic=True
            )
    value = orthopy.line.evaluate_orthogonal_polynomial(t, p0, a, b, c)

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
    p0, a, b, c = orthopy.line.recurrence_coefficients.legendre(
            n, 'monic', symbolic=True
            )
    value = orthopy.line.evaluate_orthogonal_polynomial(t, p0, a, b, c)

    assert (value == ref).all()

    # Evaluating the Legendre polynomial in this way is rather unstable, so
    # don't go too far with n.
    approx_ref = numpy.polyval(legendre(n, monic=True), t)
    assert (abs(value - approx_ref) < tol).all()
    return


def test_clenshaw(tol=1.0e-14):
    n = 5
    _, _, alpha, beta = \
        orthopy.line.recurrence_coefficients.legendre(n, 'monic')
    t = 1.0

    a = numpy.ones(n+1)
    value = orthopy.line.clenshaw(a, alpha, beta, t)

    ref = math.fsum([
            numpy.polyval(legendre(i, monic=True), t)
            for i in range(n+1)])

    assert abs(value - ref) < tol
    return


@pytest.mark.skipif(
    LooseVersion(scipy.__version__) < LooseVersion('1.0.0'),
    reason='Requires SciPy 1.0'
    )
def test_gautschi_how_to_and_how_not_to():
    '''Test Gautschi's famous example from

    W. Gautschi,
    How and how not to check Gaussian quadrature formulae,
    BIT Numerical Mathematics,
    June 1983, Volume 23, Issue 2, pp 209–216,
    <https://doi.org/10.1007/BF02218441>.
    '''
    points = numpy.array([
        1.457697817613696e-02,
        8.102669876765460e-02,
        2.081434595902250e-01,
        3.944841255669402e-01,
        6.315647839882239e-01,
        9.076033998613676e-01,
        1.210676808760832,
        1.530983977242980,
        1.861844587312434,
        2.199712165681546,
        2.543839804028289,
        2.896173043105410,
        3.262066731177372,
        3.653371887506584,
        4.102376773975577,
        ])
    weights = numpy.array([
        3.805398607861561e-2,
        9.622028412880550e-2,
        1.572176160500219e-1,
        2.091895332583340e-1,
        2.377990401332924e-1,
        2.271382574940649e-1,
        1.732845807252921e-1,
        9.869554247686019e-2,
        3.893631493517167e-2,
        9.812496327697071e-3,
        1.439191418328875e-3,
        1.088910025516801e-4,
        3.546866719463253e-6,
        3.590718819809800e-8,
        5.112611678291437e-11,
        ])

    # weight function exp(-t**3/3)
    n = len(points)
    moments = numpy.array([
        3.0**((k-2)/3.0) * math.gamma((k+1) / 3.0)
        for k in range(2*n)
        ])

    alpha, beta = orthopy.line.coefficients_from_gauss(points, weights)
    # alpha, beta = orthopy.line.chebyshev(moments)

    errors_alpha, errors_beta = \
        orthopy.line.check_coefficients(moments, alpha, beta)

    assert numpy.max(errors_alpha) > 1.0e-2
    assert numpy.max(errors_beta) > 1.0e-2
    return


def test_logo():
    import matplotlib.pyplot as plt

    max_n = 6
    moments = numpy.zeros(2*max_n)
    moments[0] = 2.0 / 3.0
    moments[2] = 8.0 / 45.0
    for n in range(max_n):
        _, _, b, c = orthopy.line.recurrence_coefficients.legendre(
            2*n, standardization='p(1)=1'
            )
        alpha, beta = \
            orthopy.line.chebyshev_modified(moments[:2*n], b, c)
        orthopy.line.plot(1, len(alpha)*[1], alpha, beta, -1.0, +1.0)

    plt.xlim(-1, +1)
    plt.ylim(-2, +2)
    plt.grid()
    plt.tick_params(
            axis='both',
            which='both',
            left='off',
            labelleft='off',
            bottom='off',
            labelbottom='off',
            )
    plt.gca().set_aspect(0.25)
    plt.show()
    # plt.savefig('logo.png', transparent=True)
    return


def test_show():
    n = 6
    moments = numpy.zeros(2*n)
    moments[0] = 2.0 / 3.0
    moments[2] = 8.0 / 45.0
    _, _, b, c = \
        orthopy.line.recurrence_coefficients.legendre(2*n, 'monic')
    alpha, beta = orthopy.line.chebyshev_modified(moments[:2*n], b, c)
    orthopy.line.show(1, len(alpha)*[1], alpha, beta, -1.0, +1.0)
    return


def test_compute_moments():
    moments = orthopy.line.compute_moments(lambda x: 1, -1, +1, 5)
    assert (
        moments == [2, 0, sympy.S(2)/3, 0, sympy.S(2)/5]
        ).all()

    moments = orthopy.line.compute_moments(
            lambda x: 1, -1, +1, 5,
            polynomial_class=orthopy.line.legendre
            )
    assert (moments == [2, 0, 0, 0, 0]).all()

    # Example from Gautschi's "How to and how not to" article
    moments = orthopy.line.compute_moments(
            lambda x: sympy.exp(-x**3/3),
            0, sympy.oo,
            5
            )

    reference = [
        3**sympy.S(n-2) / 3 * sympy.gamma(sympy.S(n+1) / 3)
        for n in range(5)
        ]

    assert numpy.all([
        sympy.simplify(m - r) == 0
        for m, r in zip(moments, reference)
        ])
    return


def test_stieltjes():
    alpha0, beta0 = orthopy.line.stieltjes(lambda t: 1, -1, +1, 5)
    _, _, alpha1, beta1 = \
        orthopy.line.recurrence_coefficients.legendre(5, 'monic')
    assert (alpha0 == alpha1).all()
    assert (beta0 == beta1).all()
    return


# def test_expt3():
#     '''Full example from Gautschi's "How to and how not to" article.
#     '''
#     # moments = orthopy.line.compute_moments(
#     #         lambda x: sympy.exp(-x**3/3),
#     #         0, sympy.oo,
#     #         31
#     #         )
#     # print(moments)
#     # alpha, beta = orthopy.line.chebyshev(moments)
#
#     alpha, beta = orthopy.line.stieltjes(
#             lambda x: sympy.exp(-x**3/3),
#             0, sympy.oo,
#             5
#             )
#     print(alpha)
#     print(beta)
#     return


@pytest.mark.parametrize(
    'k', [0, 2, 4]
    )
def test_xk(k):
    n = 10

    moments = orthopy.line.compute_moments(lambda x: x**k, -1, +1, 2*n)
    alpha, beta = orthopy.line.chebyshev(moments)

    assert (alpha == 0).all()
    assert beta[0] == moments[0]
    assert beta[1] == sympy.S(k+1) / (k+3)
    assert beta[2] == sympy.S(4) / ((k+5) * (k+3))
    orthopy.line.schemes.custom(
        numpy.array([sympy.N(a) for a in alpha], dtype=float),
        numpy.array([sympy.N(b) for b in beta], dtype=float),
        mode='numpy'
        )

    # a, b = \
    #     orthopy.line.recurrence_coefficients.legendre(
    #             2*n, mode='sympy'
    #             )

    # moments = orthopy.line.compute_moments(
    #         lambda x: x**2, -1, +1, 2*n,
    #         polynomial_class=orthopy.line.legendre
    #         )
    # alpha, beta = orthopy.line.chebyshev_modified(moments, a, b)
    # points, weights = orthopy.line.schemes.custom(
    #         numpy.array([sympy.N(a) for a in alpha], dtype=float),
    #         numpy.array([sympy.N(b) for b in beta], dtype=float)
    #         )
    return


if __name__ == '__main__':
    # test_gauss('mpmath')
    test_logo()
