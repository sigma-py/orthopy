# -*- coding: utf-8 -*-
#
import numpy
from scipy.linalg.lapack import get_lapack_funcs

from . import recurrence_coefficients

from ..tools import line_tree


def clenshaw(a, alpha, beta, t):
    '''Clenshaw's algorithm for evaluating

    S(t) = \\sum a_k P_k(alpha, beta)(t)

    where P_k(alpha, beta) is the kth orthogonal polynomial defined by the
    recurrence coefficients alpha, beta.

    See <https://en.wikipedia.org/wiki/Clenshaw_algorithm> for details.
    '''
    n = len(alpha)
    assert len(beta) == n
    assert len(a) == n + 1

    try:
        b = numpy.empty((n+1,) + t.shape)
    except AttributeError:  # 'float' object has no attribute 'shape'
        b = numpy.empty(n+1)

    # b[0] is unused, can be any value
    # TODO shift the array
    b[0] = 1.0

    b[n] = a[n]
    b[n-1] = a[n-1] + (t - alpha[n-1]) * b[n]
    for k in range(n-2, 0, -1):
        b[k] = a[k] + (t - alpha[k]) * b[k+1] - beta[k+1] * b[k+2]

    phi0 = 1.0
    phi1 = t - alpha[0]

    return phi0 * a[0] + phi1 * b[1] - beta[1] * phi0 * b[2]


def tree_legendre(n, standardization, X, symbolic=False):
    args = recurrence_coefficients.legendre(
            n, standardization, symbolic=symbolic
            )
    return line_tree(X, *args)


# pylint: disable=too-many-arguments
def tree_jacobi(n, alpha, beta, standardization, X, symbolic=False):
    args = recurrence_coefficients.jacobi(
            n, alpha, beta, standardization, symbolic=symbolic
            )
    return line_tree(X, *args)


def evaluate_orthogonal_polynomial(t, p0, a, b, c):
    '''Evaluate the orthogonal polynomial defined by its recurrence coefficients
    a, b, and c at the point(s) t.
    '''
    vals1 = numpy.zeros_like(t, dtype=int)
    # The order is important here; see
    # <https://github.com/sympy/sympy/issues/13637>.
    vals2 = numpy.ones_like(t) * p0

    for a_k, b_k, c_k in zip(a, b, c):
        vals0, vals1 = vals1, vals2
        vals2 = vals1 * (t*a_k - b_k) - vals0 * c_k
    return vals2


def show(*args, **kwargs):
    import matplotlib.pyplot as plt
    plot(*args, **kwargs)
    plt.show()
    return


# pylint: disable=too-many-arguments
def plot(p0, a, b, c, t0, t1):
    import matplotlib.pyplot as plt
    n = 1000
    t = numpy.linspace(t0, t1, n)
    vals = evaluate_orthogonal_polynomial(t, p0, a, b, c)
    plt.plot(t, vals)
    return
