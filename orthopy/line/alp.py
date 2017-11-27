# -*- coding: utf-8 -*-
#
from __future__ import division

import numpy
import sympy


# pylint: disable=too-many-arguments
def alp_tree(
        n, x, phi=None, normalization=None,
        with_condon_shortley_phase=True
        ):
    '''Evaluates the entire tree of associated Legendre polynomials up to depth
    n.
    There are many recurrence relations that can be used to construct the
    associated Legendre polynomials. However, only few are numerically stable.
    Many implementations (including this one) use the classical Legendre
    recurrence relation with increasing L.

    Useful references are

    Taweetham Limpanuparb, Josh Milthorpe,
    Associated Legendre Polynomials and Spherical Harmonics Computation for
    Chemistry Applications,
    Proceedings of The 40th Congress on Science and Technology of Thailand;
    2014 Dec 2-4, Khon Kaen, Thailand. P. 233-241.
    <https://arxiv.org/abs/1410.1748>

    and

    Schneider et al.,
    A new Fortran 90 program to compute regular and irregular associated
    Legendre functions,
    Computer Physics Communications,
    Volume 181, Issue 12, December 2010, Pages 2091-2097,
    <https://doi.org/10.1016/j.cpc.2010.08.038>.

    The return value is a list of arrays, where `out[k]` hosts the `2*k+1`
    values of the `k`th level of the tree

                              (0, 0)
                    (-1, 1)   (0, 1)   (1, 1)
          (-2, 2)   (-1, 2)   (0, 2)   (1, 2)   (2, 2)
            ...       ...       ...     ...       ...
    '''
    # pylint: disable=too-many-statements,too-many-locals
    assert numpy.all(numpy.abs(x) <= 1.0)

    sqrt = numpy.vectorize(sympy.sqrt)
    sqrt1mx2 = sqrt(1 - x**2)

    e = numpy.ones_like(x, dtype=int)

    if normalization is None:
        alpha = 1
        out = [[alpha * e]]

        def z0_factor(L):
            return sqrt1mx2 / (2*L)

        def z1_factor(L):
            return sqrt1mx2 * (2*L-1)

        def C0(L):
            m = numpy.arange(-L+1, L)
            return (2*L-1) / (L-m)

        def C1(L):
            m = numpy.arange(-L+2, L-1)
            return (L-1+m) / (L-m)

    elif normalization == 'spherical':
        alpha = 1.0 / 2 / numpy.sqrt(numpy.pi)
        out = [[alpha * e]]

        def z0_factor(L):
            return numpy.sqrt((2*L+1) / (2*L)) * sqrt1mx2

        def z1_factor(L):
            return numpy.sqrt((2*L+1) / (2*L)) * sqrt1mx2

        def C0(L):
            m = numpy.arange(-L+1, L)
            d = (L+m) * (L-m) / (2*L+1)
            return numpy.sqrt((2*L-1) / d)

        def C1(L):
            m = numpy.arange(-L+2, L-1)
            d = (L+m) * (L-m) / (2*L+1)
            return numpy.sqrt((L+m-1) * (L-m-1) / (2*L-3) / d)
    elif normalization == 'complex spherical':
        alpha = 1.0 / 2 / numpy.sqrt(numpy.pi)
        out = [[alpha * numpy.ones_like(x, dtype=complex)]]

        def z0_factor(L):
            return numpy.sqrt((2*L+1)/(2*L)) * sqrt1mx2 * numpy.exp(-1j * phi)

        def z1_factor(L):
            return numpy.sqrt((2*L+1)/(2*L)) * sqrt1mx2 * numpy.exp(+1j * phi)

        def C0(L):
            m = numpy.arange(-L+1, L)
            d = (L+m) * (L-m) / (2*L+1)
            return numpy.sqrt((2*L-1) / d)

        def C1(L):
            m = numpy.arange(-L+2, L-1)
            d = (L+m) * (L-m) / (2*L+1)
            return numpy.sqrt((L+m-1) * (L-m-1) / (2*L-3) / d)
    elif normalization == 'full':
        alpha = numpy.sqrt(0.5)
        out = [[alpha * e]]

        def z0_factor(L):
            return numpy.sqrt((2*L+1) / (2*L)) * sqrt1mx2

        def z1_factor(L):
            return numpy.sqrt((2*L+1) / (2*L)) * sqrt1mx2

        def C0(L):
            m = numpy.arange(-L+1, L)
            d = (L+m) * (L-m) / (2*L+1)
            return numpy.sqrt((2*L-1) / d)

        def C1(L):
            m = numpy.arange(-L+2, L-1)
            d = (L+m) * (L-m) / (2*L+1)
            return numpy.sqrt((L+m-1) * (L-m-1) / (2*L-3) / d)
    else:
        assert normalization == 'schmidt', \
            'Unknown normalization \'{}\'.'.format(normalization)

        alpha = 2
        out = [[alpha * e]]

        def z0_factor(L):
            return sqrt1mx2 * sqrt(sympy.Rational(2*L-1, 2*L))

        def z1_factor(L):
            return sqrt1mx2 * sqrt(sympy.Rational(2*L-1, 2*L))

        def C0(L):
            m = numpy.arange(-L+1, L)
            d = sqrt((L+m) * (L-m))
            return (2*L-1) / d

        def C1(L):
            m = numpy.arange(-L+2, L-1)
            d = sqrt((L+m) * (L-m))
            return sqrt((L+m-1) * (L-m-1)) / d

    if with_condon_shortley_phase:
        def z1_factor_CSP(L):
            return -1 * z1_factor(L)
    else:
        z1_factor_CSP = z1_factor

    # Here comes the actual loop.
    for L in range(1, n+1):
        out.append(
            numpy.concatenate([
                [out[L-1][0] * z0_factor(L)],
                out[L-1] * numpy.multiply.outer(C0(L), x),
                [out[L-1][-1] * z1_factor_CSP(L)],
                ])
            )

        if L > 1:
            out[-1][2:-2] -= numpy.multiply.outer(C1(L), e) * out[L-2]

    return out
