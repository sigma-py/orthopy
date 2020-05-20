import itertools

import sympy

from ..tools import Iterator1D
from . import recurrence_coefficients


def tree_legendre(n, X, standardization, symbolic=False):
    return tree_gegenbauer(n, X, 0, standardization, symbolic=symbolic)


def tree_chebyshev2(n, X, standardization, symbolic=False):
    one_half = sympy.S(1) / 2 if symbolic else 0.5
    return tree_gegenbauer(n, X, +one_half, standardization, symbolic=symbolic)


def tree_gegenbauer(n, X, lmbda, standardization, symbolic=False):
    return tree_jacobi(n, X, lmbda, lmbda, standardization, symbolic=symbolic)


def tree_jacobi(n, *args, **kwargs):
    return list(itertools.islice(Jacobi(*args, **kwargs), n + 1))


class Jacobi(Iterator1D):
    def __init__(self, X, *args, **kwargs):
        super().__init__(X, recurrence_coefficients.Jacobi(*args, **kwargs))
