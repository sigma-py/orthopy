import itertools

import numpy
import pytest
import sympy

import orthopy

X = sympy.symbols("x, y")


# def test_zernike():
#     p = [sympy.poly(x, X) for x in X]
#     vals = numpy.concatenate(orthopy.s2.yu.tree(3, p, symbolic=True))
#     vals[0] = sympy.poly(vals[0], X)
#
#     for val in vals:
#         print(val)
#
#     # https://en.wikipedia.org/wiki/Zernike_polynomials
#     r = [
#         {0: lambda r: 1},
#         {1: lambda r: r},
#         {0: lambda r: 2 * r ** 2 - 1, 2: lambda r: r ** 2}
#     ]
#
#     zmn = r[m][n] * cos(m * phi)
#
#     exit(1)
#     return


# def _integrate(f):
#     # Cartesian integration in sympy is bugged, cf.
#     # <https://github.com/sympy/sympy/issues/13816>.
#     # Simply transform to polar coordinates for now.
#     r = sympy.Symbol("r")
#     phi = sympy.Symbol("phi")
#     return sympy.integrate(
#         r * f.subs([(X[0], r * sympy.cos(phi)), (X[1], r * sympy.sin(phi))]),
#         (r, 0, 1),
#         (phi, 0, 2 * sympy.pi),
#     )


def volume_nball(n, symbolic, r=1):
    pi = sympy.pi if symbolic else numpy.pi

    if n == 0:
        return 1
    elif n == 1:
        return 2 * r
    return volume_nball(n - 2, symbolic, r=r) * 2 * pi / n * r ** 2


def _integrate_monomial(k, symbolic=True, r=1):
    frac = sympy.Rational if symbolic else lambda a, b: a / b
    if any(a % 2 == 1 for a in k):
        return 0

    n = len(k)
    if all(a == 0 for a in k):
        return volume_nball(n, symbolic, r=r)

    # find first nonzero
    idx = next(i for i, j in enumerate(k) if j > 0)
    alpha = frac((k[idx] - 1) * r ** 2, sum(k) + n)
    k2 = k.copy()
    k2[idx] -= 2
    return _integrate_monomial(k2, symbolic, r=r) * alpha


def _integrate_poly(p):
    return sum(c * _integrate_monomial(list(k)) for c, k in zip(p.coeffs(), p.monoms()))


@pytest.mark.parametrize(
    "scaling,int0",
    [("classical", sympy.pi), ("monic", sympy.pi), ("normal", sympy.sqrt(sympy.pi))],
)
def test_yu_integral0(scaling, int0, n=4):
    p = [sympy.poly(x, X) for x in X]
    vals = numpy.concatenate(orthopy.s2.yu.tree(n, p, scaling, symbolic=True))
    vals[0] = sympy.poly(vals[0], X)

    assert _integrate_poly(vals[0]) == int0
    for val in vals[1:]:
        assert _integrate_poly(val) == 0


@pytest.mark.parametrize("scaling", ["classical", "monic", "normal"])
def test_yu_orthogonality(scaling, n=4):
    p = [sympy.poly(x, X) for x in X]
    tree = numpy.concatenate(orthopy.s2.yu.tree(n, p, scaling, symbolic=True))
    vals = tree * numpy.roll(tree, 1, axis=0)

    for val in vals:
        assert _integrate_poly(val) == 0


def test_yu_normality(n=4):
    p = [sympy.poly(x, X) for x in X]
    iterator = orthopy.s2.yu.Eval(p, "normal", symbolic=True)
    for k, vals in enumerate(itertools.islice(iterator, n)):
        if k == 0:
            vals[0] = sympy.poly(vals[0], X)
        for val in vals:
            assert _integrate_poly(val ** 2) == 1


def test_zernike_vals():
    from math import factorial as fact

    # explicit formula for R_n^m
    def r(n, m, rho):
        assert n >= m >= 0
        assert (n - m) % 2 == 0
        s = 0.0
        for k in range((n - m) // 2 + 1):
            s += (
                (-1) ** k
                * fact(n - k)
                / fact(k)
                / fact((n + m) // 2 - k)
                / fact((n - m) // 2 - k)
                * rho ** (n - 2 * k)
            )
        return s

    numpy.random.seed(0)
    rho, phi = numpy.random.rand(2)

    xy = [rho * numpy.cos(phi), rho * numpy.sin(phi)]
    iterator = orthopy.s2.zernike.Eval(xy, "classical")
    for n, vals in enumerate(itertools.islice(iterator, 5)):
        for m, val in zip(range(-n, n + 1, 2), vals):
            fun = numpy.sin if m < 0 else numpy.cos
            ref = r(n, abs(m), rho) * fun(abs(m) * phi)
            assert abs(ref - val) < 1.0e-12 * abs(ref)


def test_zernike_explicit():

    X = sympy.symbols("x, y")
    x, y = X
    ref = [
        [1],
        [y, x],
        [2 * x * y, 2 * (x ** 2 + y ** 2) - 1, x ** 2 - y ** 2],
        [
            3 * x ** 2 * y - y ** 3,
            3 * (x ** 2 + y ** 2) * y - 2 * y,
            3 * (x ** 2 + y ** 2) * x - 2 * x,
            x ** 3 - 3 * y ** 2 * x,
        ],
        [
            4 * y * x ** 3 - 4 * y ** 3 * x,
            (4 * (x ** 2 + y ** 2) - 3) * 2 * x * y,
            6 * (x ** 2 + y ** 2) ** 2 - 6 * (x ** 2 + y ** 2) + 1,
            (4 * (x ** 2 + y ** 2) - 3) * (x ** 2 - y ** 2),
            x ** 4 - 6 * x ** 2 * y ** 2 + y ** 4,
        ],
        [
            5 * y * x ** 4 - 10 * y ** 3 * x ** 2 + y ** 5,
            (5 * (x ** 2 + y ** 2) - 4) * (3 * y * x ** 2 - y ** 3),
            (10 * (x ** 2 + y ** 2) ** 2 - 12 * (x ** 2 + y ** 2) + 3) * y,
            (10 * (x ** 2 + y ** 2) ** 2 - 12 * (x ** 2 + y ** 2) + 3) * x,
            (5 * (x ** 2 + y ** 2) - 4) * (x ** 3 - 3 * y ** 2 * x),
            x ** 5 - 10 * y ** 2 * x ** 3 + 5 * y ** 4 * x,
        ]
    ]
    iterator = orthopy.s2.zernike.Eval(X, "classical", symbolic=True)
    for ref_level, vals in zip(ref, itertools.islice(iterator, 6)):
        for r, val in zip(ref_level, vals):
            # r = sympy.simplify(r)
            # val = sympy.simplify(val)
            assert sympy.simplify(r - val) == 0, f"ref = {r}  !=   {val}"


# @pytest.mark.parametrize("scaling,int0", [("classical", sympy.pi)])
# def test_zernike_integral0(scaling, int0, n=4):
#     p = [sympy.poly(x, X) for x in X]
#     vals = numpy.concatenate(orthopy.s2.zernike.tree(n, p, scaling, symbolic=True))
#     vals[0] = sympy.poly(vals[0], X)
#
#     assert _integrate_poly(vals[0]) == int0
#     for val in vals[1:]:
#         assert _integrate_poly(val) == 0


@pytest.mark.parametrize("scaling", ["classical"])
def test_zernike_orthogonality(scaling, n=4):
    p = [sympy.poly(x, X) for x in X]
    tree = numpy.concatenate(orthopy.s2.zernike.tree(n, p, scaling, symbolic=True))
    vals = tree * numpy.roll(tree, 1, axis=0)

    for val in vals:
        assert _integrate_poly(val) == 0


# def test_zernike_normality(n=4):
#     p = [sympy.poly(x, X) for x in X]
#     iterator = orthopy.s2.zernike.Eval(p, "normal", symbolic=True)
#     for k, vals in enumerate(itertools.islice(iterator, n)):
#         if k == 0:
#             vals[0] = sympy.poly(vals[0], X)
#         for val in vals:
#             assert _integrate_poly(val ** 2) == 1


def test_show(scaling="normal", n=2, r=1):
    def f(X):
        return orthopy.s2.zernike.tree(n, X, scaling)[n][r]

    # k = numpy.linspace(0, 2 * numpy.pi, 100000)
    # X = numpy.array([numpy.cos(k), numpy.sin(k)])
    # print(numpy.min(f(X)), numpy.max(f(X)))
    # x0 = [numpy.sqrt(1 / (n)), numpy.sqrt((n-1) / (n))]
    # print(f(x0))

    # p = [sympy.poly(x, X) for x in X]
    # iterator = orthopy.s2.yu.Eval(p, "classical", symbolic=True)
    # for k, vals in enumerate(itertools.islice(iterator, 10)):
    #     print()
    #     for val in vals:
    #         print(val)
    # exit(1)

    orthopy.s2.show(f, lcar=1.0e-2)
    # orthopy.s2.plot(f, lcar=2.0e-2)
    # import matplotlib.pyplot as plt
    # plt.savefig('s2.png', transparent=True)


if __name__ == "__main__":
    # p = [sympy.poly(x, X) for x in X]
    # X = [1.0, 0.0]
    # iterator = orthopy.s2.zernike.Eval(X, "classical", symbolic=False)
    # for vals in itertools.islice(iterator, 10):
    #     print(vals[0])
    # test_zernike()
    test_show("classical", n=5, r=2)
