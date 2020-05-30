import itertools

import numpy
import sympy

import orthopy

X = sympy.symbols("x, y")


# def test_zernicke():
#     p = [sympy.poly(x, X) for x in X]
#     vals = numpy.concatenate(orthopy.s2.tree(3, p, symbolic=True))
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


def test_integral0(n=4):
    p = [sympy.poly(x, X) for x in X]
    vals = numpy.concatenate(orthopy.s2.tree(n, p, symbolic=True))
    vals[0] = sympy.poly(vals[0], X)

    assert _integrate_poly(vals[0]) == sympy.sqrt(sympy.pi)
    for val in vals[1:]:
        assert _integrate_poly(val) == 0


def test_orthogonality(n=4):
    p = [sympy.poly(x, X) for x in X]
    tree = numpy.concatenate(orthopy.s2.tree(n, p, symbolic=True))
    vals = tree * numpy.roll(tree, 1, axis=0)

    for val in vals:
        assert _integrate_poly(val) == 0


def test_normality(n=4):
    p = [sympy.poly(x, X) for x in X]
    iterator = orthopy.s2.Iterator(p, symbolic=True)
    for k, vals in enumerate(itertools.islice(iterator, n)):
        if k == 0:
            vals[0] = sympy.poly(vals[0], X)
        for val in vals:
            assert _integrate_poly(val ** 2) == 1


def test_show(n=2, r=1):
    def f(X):
        return orthopy.s2.tree(n, X)[n][r]

    orthopy.s2.show(f)
    # orthopy.s2.plot(f, lcar=2.0e-2)
    # import matplotlib.pyplot as plt
    # plt.savefig('s2.png', transparent=True)


if __name__ == "__main__":
    test_zernicke()
