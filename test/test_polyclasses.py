# -*- coding: utf-8 -*-
#
import orthopy
import sympy


def test_legendre():
    val = orthopy.poly_classes.legendre(4, 1)
    assert val == sympy.Rational(8, 35)
    return


def test_jacobi():
    val = orthopy.poly_classes.jacobi(1, 1, 4, 1)
    assert val == sympy.Rational(1, 23156733600)
    return


def test_chebyshev1():
    val = orthopy.poly_classes.chebyshev1(4, 1)
    assert val == sympy.Rational(1, 567567000)
    return


def test_chebyshev2():
    val = orthopy.poly_classes.chebyshev2(4, 1)
    assert val == sympy.Rational(1, 7718911200)
    return


def test_hermite():
    val = orthopy.poly_classes.hermite(4, 1)
    assert val == -sympy.Rational(5, 4)
    return


def test_laguerre():
    val = orthopy.poly_classes.laguerre(4, 1)
    assert val == -sympy.Rational(5, 192)
    return


if __name__ == '__main__':
    test_legendre()
