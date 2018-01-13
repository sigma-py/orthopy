# -*- coding: utf-8 -*-
#
import matplotlib.pyplot as plt
import numpy

from .orth import tree_jacobi


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


def show(*args, **kwargs):
    plot(*args, **kwargs)
    plt.show()
    return


def plot(L, alpha, beta):
    xlim = [-1.0, +1.0]
    x = numpy.linspace(xlim[0], xlim[1], 500)
    vals = tree_jacobi(L, alpha, beta, 'normal', x)

    for val in vals:
        plt.plot(x, val)

    # plt.axes().set_aspect('equal')

    plt.xlim(*xlim)
    # plt.ylim(-2, +2)
    plt.tick_params(
        axis='both',
        which='both',
        bottom='off',
        top='off',
        left='off',
        right='off',
        labelbottom='off',
        labelleft='off'
        )
    plt.grid()
    return
