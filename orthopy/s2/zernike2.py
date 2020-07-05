import itertools
import math

import numpy
import sympy

from .tools import plot_single as ps


def tree(n, *args, **kwargs):
    return list(itertools.islice(Eval(*args, **kwargs), n + 1))


def savefig_single(filename, *args, **kwargs):
    import matplotlib.pyplot as plt

    plot_single(*args, **kwargs)
    plt.savefig(filename, transparent=True, bbox_inches="tight")


def show_single(*args, **kwargs):
    import matplotlib.pyplot as plt

    plot_single(*args, **kwargs)
    plt.show()


def plot_single(*args, **kwargs):
    ps("Zernike-2", Eval, *args, **kwargs)


def savefig_tree(filename, *args, **kwargs):
    import matplotlib.pyplot as plt

    plot_tree(*args, **kwargs)
    plt.savefig(filename, transparent=True, bbox_inches="tight")


def show_tree(*args, **kwargs):
    import matplotlib.pyplot as plt

    plot_tree(*args, **kwargs)
    plt.show()


def plot_tree(*args, **kwargs):
    from .tools import plot_tree as pt

    pt("Zernike-2", Eval, *args, **kwargs)


class Eval:
    """
    Similar to regular Zernike, but a lot simpler. Can probably be generalized to
    n-ball.
    """

    def __init__(self, X, scaling, symbolic=False):
        self.rc = {"classical": RCClassical, "monic": RCMonic, "normal": RCNormal}[
            scaling
        ](symbolic)
        self.X = X
        self.L = 0
        self.last = [None, None]

    def __iter__(self):
        return self

    def __next__(self):
        if self.L == 0:
            out = numpy.array([0 * self.X[0] + self.rc.p0])
        else:
            alpha, beta, gamma = self.rc[self.L]

            shape = list(self.last[0].shape)
            shape[0] += 1
            out = numpy.zeros(shape, dtype=self.last[0].dtype)

            last_X = self.last[0] * self.X[0]
            last_Y = self.last[0] * self.X[1]

            # The minus sign could go onto the other last_Y, too.
            out[1:] += alpha * (last_X + last_Y[::-1])
            out[:-1] += beta * (last_X - last_Y[::-1])
            if self.L > 1:
                out[1:-1] -= gamma * self.last[1]

        self.last[1] = self.last[0]
        self.last[0] = out
        self.L += 1
        return out


class RCClassical:
    def __init__(self, symbolic):
        self.p0 = 1

    def __getitem__(self, n):
        assert n > 0

        alpha = 1
        beta = 1
        if n == 1:
            gamma = None
        else:
            gamma = 1
        return alpha, beta, gamma


class RCMonic:
    def __init__(self, symbolic):
        self.S = sympy.S if symbolic else lambda x: x
        self.p0 = 1

    def __getitem__(self, n):
        assert n > 0

        n = self.S(n)
        alpha = numpy.array([i / n for i in range(1, n + 1)])
        beta = numpy.array([(n - i) / n for i in range(n)])
        if n == 1:
            gamma = None
        else:
            gamma = numpy.array([i * (n - i) / (n * (n - 1)) for i in range(1, n)])
        return alpha, beta, gamma


class RCNormal:
    def __init__(self, symbolic):
        self.S = sympy.S if symbolic else lambda x: x
        self.sqrt = sympy.sqrt if symbolic else math.sqrt
        pi = sympy.pi if symbolic else math.pi
        self.p0 = 1 / self.sqrt(pi)

    def __getitem__(self, n):
        assert n > 0
        n = self.S(n)
        sqrt = self.sqrt

        alpha = sqrt((n + 1) / n)
        beta = sqrt((n + 1) / n)
        if n == 1:
            gamma = None
        else:
            gamma = sqrt((n + 1) / (n - 1))
        return alpha, beta, gamma
