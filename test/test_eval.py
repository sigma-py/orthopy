# -*- coding: utf-8 -*-
#
import numpy
import orthopy
import pytest
from sympy import Rational, sqrt


@pytest.mark.parametrize('n, val0, val1, val2', [
    (0, 1, 1, 1),
    (1, 0, Rational(1, 2), 1),
    (2, -Rational(1, 3), -Rational(1, 12), Rational(2, 3)),
    (3, 0, -Rational(7, 40), Rational(2, 5)),
    (4, Rational(3, 35), -Rational(37, 560), Rational(8, 35)),
    (5, 0, Rational(23, 2016), Rational(8, 63)),
    ]
    )
def test_legendre_monic(n, val0, val1, val2):
    # Test evaluation of one value
    val = orthopy.eval.legendre(n, 0)
    assert val == val0

    # Test evaluation of multiple values
    x = numpy.array([Rational(1, 2), 1])
    val = orthopy.eval.legendre(n, x)
    print(val)
    print(val1, val2)
    assert all(val == numpy.array([val1, val2]))
    return


@pytest.mark.parametrize('n, val0, val1, val2', [
    (0, 1, 1, 1),
    (1, 0, Rational(1, 2), 1),
    (2, -Rational(1, 2), -Rational(1, 8), 1),
    (3, 0, -Rational(7, 16), 1),
    (4, Rational(3, 8), -Rational(37, 128), 1),
    (5, 0, Rational(23, 256), 1),
    ]
    )
def test_legendre_p11(n, val0, val1, val2):
    # Test evaluation of one value
    val = orthopy.eval.legendre(n, 0, normalization='p(1)=1')
    assert val == val0

    # Test evaluation of multiple values
    x = numpy.array([Rational(1, 2), 1])
    val = orthopy.eval.legendre(n, x, normalization='p(1)=1')
    assert all(val == numpy.array([val1, val2]))
    return


@pytest.mark.parametrize('n, val0, val1, val2', [
    (0, sqrt(Rational(1, 2)), sqrt(Rational(1, 2)), sqrt(Rational(1, 2))),
    (1, 0, sqrt(Rational(3, 8)), sqrt(Rational(3, 2))),
    (2, -sqrt(Rational(5, 8)), -sqrt(Rational(5, 128)), sqrt(Rational(5, 2))),
    (3, 0, -sqrt(Rational(343, 512)), sqrt(Rational(7, 2))),
    (4, 9 / sqrt(2) / 8, -111 / sqrt(2) / 128, 3/sqrt(2)),
    (5, 0, sqrt(Rational(5819, 131072)), sqrt(Rational(11, 2))),
    ]
    )
def test_legendre_pnorm1(n, val0, val1, val2):
    # Test evaluation of one value
    val = orthopy.eval.legendre(n, 0, normalization='||p**2||=1')
    assert val == val0

    # Test evaluation of multiple values
    x = numpy.array([Rational(1, 2), 1])
    val = orthopy.eval.legendre(n, x, normalization='||p**2||=1')
    assert all(val == numpy.array([val1, val2]))
    return


if __name__ == '__main__':
    test_legendre_monic(0, 1, 1, 1)
