import sympy

from ..tools import line_tree
from . import recurrence_coefficients


def tree_chebyshev1(X, n, standardization, symbolic=False):
    one_half = sympy.S(1) / 2 if symbolic else 0.5
    return tree_jacobi(X, n, -one_half, -one_half, standardization, symbolic=symbolic)


def tree_chebyshev2(X, n, standardization, symbolic=False):
    one_half = sympy.S(1) / 2 if symbolic else 0.5
    return tree_jacobi(X, n, +one_half, +one_half, standardization, symbolic=symbolic)


def tree_legendre(X, n, standardization, symbolic=False):
    return tree_jacobi(X, n, 0, 0, standardization, symbolic=symbolic)


def tree_gegenbauer(X, n, lmbda, standardization, symbolic=False):
    return tree_jacobi(X, n, lmbda, lmbda, standardization, symbolic=symbolic)


def tree_jacobi(X, n, alpha, beta, standardization, symbolic=False):
    args = recurrence_coefficients.jacobi(
        n, alpha, beta, standardization, symbolic=symbolic
    )
    return line_tree(X, *args)
