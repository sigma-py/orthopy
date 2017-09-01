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


if __name__ == '__main__':
    test_legendre()
