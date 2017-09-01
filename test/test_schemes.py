# -*- coding: utf-8 -*-
#
from mpmath import mp
import orthopy


def test_legendre():
    points, weights = orthopy.schemes.legendre(4, decimal_places=50)

    tol = 1.0e-50

    x1 = mp.sqrt(mp.mpf(3)/7 - mp.mpf(2)/7 * mp.sqrt(mp.mpf(6)/5))
    x2 = mp.sqrt(mp.mpf(3)/7 + mp.mpf(2)/7 * mp.sqrt(mp.mpf(6)/5))
    assert (abs(points - [-x2, -x1, +x1, +x2]) < tol).all()

    w1 = (18 + mp.sqrt(30)) / 36
    w2 = (18 - mp.sqrt(30)) / 36
    assert (abs(weights - [w2, w1, w1, w2]) < tol).all()
    return


def test_chebyshev1():
    points, weights = orthopy.schemes.chebyshev1(4, decimal_places=51)

    tol = 1.0e-50

    x1 = mp.cos(3 * mp.pi/8)
    x2 = mp.cos(1 * mp.pi/8)
    assert (abs(points - [-x2, -x1, +x1, +x2]) < tol).all()

    w = mp.pi / 4
    tol = 1.0e-50
    assert (abs(weights - [w, w, w, w]) < tol).all()
    return


def test_chebyshev2():
    points, weights = orthopy.schemes.chebyshev2(4, decimal_places=51)

    tol = 1.0e-50

    x1 = mp.cos(2 * mp.pi/5)
    x2 = mp.cos(1 * mp.pi/5)
    assert (abs(points - [-x2, -x1, +x1, +x2]) < tol).all()

    w1 = mp.pi / 5 * mp.sin(2 * mp.pi/5)**2
    w2 = mp.pi / 5 * mp.sin(1 * mp.pi/5)**2
    assert (abs(weights - [w2, w1, w1, w2]) < tol).all()
    return


if __name__ == '__main__':
    test_chebyshev2()
