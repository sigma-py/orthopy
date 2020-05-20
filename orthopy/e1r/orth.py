import itertools
import math

import numpy
import sympy

from ..tools import Iterator1D


def tree(X, n, **kwargs):
    return list(itertools.islice(Iterator(X, **kwargs), n + 1))


class Iterator:
    def __init__(self, X, **kwargs):
        self.iterator1d = Iterator1D(X, IteratorRC(**kwargs))

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.iterator1d)


class IteratorRC:
    """Recurrence coefficients for generalized Laguerre polynomials. Set alpha=0
    (default) to get classical Laguerre.
    """

    def __init__(self, alpha=0, standardization="normal", symbolic=False):
        self.sqrt = sympy.sqrt if symbolic else numpy.sqrt
        self.gamma = sympy.gamma if symbolic else math.gamma
        self.S = sympy.S if symbolic else lambda a: a

        self.alpha = alpha
        self.standardization = standardization

        if standardization == "monic":
            self.p0 = 1
        elif standardization == "classical":
            self.p0 = 1
        else:
            assert (
                standardization == "normal"
            ), "Unknown Laguerre standardization '{}'.".format(standardization)
            self.p0 = 1 / self.sqrt(self.gamma(alpha + 1))

        self.k = 0

    def __iter__(self):
        return self

    def __next__(self):
        gamma = self.gamma
        sqrt = self.sqrt
        alpha = self.alpha
        S = self.S
        k = self.k

        if self.standardization == "monic":
            a = 1
            b = 2 * k + 1 + alpha
            # TODO remove specialization for k == 0
            if self.k == 0:
                c = gamma(alpha + 1)
            else:
                c = k * (k + alpha)
        elif self.standardization == "classical":
            a = -S(1) / (k + 1)
            b = -S(2 * k + 1 + alpha) / (k + 1)
            # TODO remove specialization for k == 0
            if self.k == 0:
                c = numpy.nan
            else:
                c = S(k + alpha) / (k + 1)
        else:
            assert self.standardization == "normal"
            a = -1 / sqrt((k + 1) * (k + 1 + alpha))
            b = -(2 * k + 1 + alpha) / sqrt((k + 1) * (k + 1 + alpha))
            # TODO remove specialization for k == 0
            if self.k == 0:
                c = numpy.nan
            else:
                c = sqrt(k * S(k + alpha) / ((k + 1) * (k + 1 + alpha)))

        self.k += 1
        return a, b, c
