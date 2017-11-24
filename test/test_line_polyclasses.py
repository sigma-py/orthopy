# -*- coding: utf-8 -*-
#
import orthopy
import sympy


def test_legendre():
    val = orthopy.line_segment.poly_classes.legendre(4, 1)
    assert val == sympy.Rational(8, 35)
    return


def test_jacobi():
    val = orthopy.line_segment.poly_classes.jacobi(
            4, 1, 1, 5,
            standardization='p(1)=(n+alpha over n)'
            )
    assert val == 7985
    val = orthopy.line_segment.poly_classes.jacobi(
            4, 1, 1, 5,
            standardization='monic'
            )
    assert val == sympy.Rational(12776, 21)
    return


def test_chebyshev1():
    val = orthopy.line_segment.poly_classes.chebyshev1(
            4, 1,
            standardization='p(1)=(n+alpha over n)'
            )
    assert val == sympy.Rational(35, 128)

    val = orthopy.line_segment.poly_classes.chebyshev1(4, 1)
    assert val == sympy.Rational(1, 8)
    return


def test_chebyshev2():
    val = orthopy.line_segment.poly_classes.chebyshev2(
            4, 1,
            standardization='p(1)=(n+alpha over n)'
            )
    assert val == sympy.Rational(315, 128)

    val = orthopy.line_segment.poly_classes.chebyshev2(4, 1)
    assert val == sympy.Rational(5, 16)
    return


def test_hermite():
    val = orthopy.line_segment.poly_classes.hermite(4, 1)
    assert val == -sympy.Rational(5, 4)
    return


def test_laguerre():
    val = orthopy.line_segment.poly_classes.laguerre(4, 1)
    assert val == -sympy.Rational(5, 192)
    return


if __name__ == '__main__':
    test_legendre()
