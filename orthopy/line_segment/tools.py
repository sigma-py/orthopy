# -*- coding: utf-8 -*-
#
import numpy

from . import recurrence_coefficients

from ..tools import line_tree, line_evaluate


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
    import matplotlib.pyplot as plt
    plot(*args, **kwargs)
    plt.show()
    return


# pylint: disable=too-many-arguments
def plot(p0, a, b, c, t0, t1):
    import matplotlib.pyplot as plt
    n = 1000
    t = numpy.linspace(t0, t1, n)
    vals = line_evaluate(t, p0, a, b, c)
    plt.plot(t, vals)
    return
