import itertools
import math

import numpy
import sympy

from ..tools import Iterator1D


def tree(X, n, **kwargs):
    return list(itertools.islice(Iterator(X, **kwargs), n + 1))


class Iterator(Iterator1D):
    """Generalized Laguerre polynomials. Set alpha=0 (default) to get classical
    Laguerre.

    The first few are (for alpha=0):

    standardization == "monic":
        1
        x - 1
        x**2 - 4*x + 2
        x**3 - 9*x**2 + 18*x - 6
        x**4 - 16*x**3 + 72*x**2 - 96*x + 24
        x**5 - 25*x**4 + 200*x**3 - 600*x**2 + 600*x - 120

    standardization == "classical" or "normal"
        1
        1 - x
        x**2/2 - 2*x + 1
        -x**3/6 + 3*x**2/2 - 3*x + 1
        x**4/24 - 2*x**3/3 + 3*x**2 - 4*x + 1
        -x**5/120 + 5*x**4/24 - 5*x**3/3 + 5*x**2 - 5*x + 1

    The classical and normal standarizations differe for alpha != 0.
    """

    def __init__(self, X, *args, **kwargs):
        super().__init__(X, IteratorRC(*args, **kwargs))


class IteratorRC:
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
