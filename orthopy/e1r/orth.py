import itertools
import math

import numpy
import sympy

from ..tools import Iterator1D


def tree(n, *args, **kwargs):
    return list(itertools.islice(Iterator(*args, **kwargs), n + 1))


class Iterator(Iterator1D):
    """Generalized Laguerre polynomials. Set alpha=0 (default) to get classical
    Laguerre.

    The first few are (for alpha=0):

    scaling == "monic":
        1
        x - 1
        x**2 - 4*x + 2
        x**3 - 9*x**2 + 18*x - 6
        x**4 - 16*x**3 + 72*x**2 - 96*x + 24
        x**5 - 25*x**4 + 200*x**3 - 600*x**2 + 600*x - 120

    scaling == "classical" or "normal"
        1
        1 - x
        x**2/2 - 2*x + 1
        -x**3/6 + 3*x**2/2 - 3*x + 1
        x**4/24 - 2*x**3/3 + 3*x**2 - 4*x + 1
        -x**5/120 + 5*x**4/24 - 5*x**3/3 + 5*x**2 - 5*x + 1

    The classical and normal standarizations differ for alpha != 0.
    """

    def __init__(self, X, scaling, *args, **kwargs):
        if scaling == "monic":
            iterator = IteratorRCMonic(*args, **kwargs)
        elif scaling == "classical":
            iterator = IteratorRCClassical(*args, **kwargs)
        else:
            assert scaling == "normal", "Unknown Laguerre scaling '{}'.".format(scaling)
            iterator = IteratorRCNormal(*args, **kwargs)

        super().__init__(X, iterator)


class IteratorRCMonic:
    def __init__(self, alpha=0, symbolic=False):
        self.alpha = alpha
        self.p0 = 1
        self.k = 0

    def __iter__(self):
        return self

    def __next__(self):
        alpha = self.alpha
        k = self.k

        a = 1
        b = 2 * k + 1 + alpha
        c = k * (k + alpha)

        self.k += 1
        return a, b, c


class IteratorRCClassical:
    def __init__(self, alpha=0, symbolic=False):
        self.S = sympy.S if symbolic else lambda a: a
        self.alpha = alpha
        self.p0 = 1
        self.k = 0

    def __iter__(self):
        return self

    def __next__(self):
        alpha = self.alpha
        S = self.S
        k = self.k

        a = -S(1) / (k + 1)
        b = -S(2 * k + 1 + alpha) / (k + 1)
        c = S(k + alpha) / (k + 1)

        self.k += 1
        return a, b, c


class IteratorRCNormal:
    def __init__(self, alpha=0, symbolic=False):
        self.sqrt = sympy.sqrt if symbolic else numpy.sqrt
        gamma = sympy.gamma if symbolic else math.gamma
        self.S = sympy.S if symbolic else lambda a: a
        self.alpha = alpha
        self.p0 = 1 / self.sqrt(gamma(alpha + 1))
        self.k = 0

    def __iter__(self):
        return self

    def __next__(self):
        sqrt = self.sqrt
        S = self.S
        alpha = self.alpha
        k = self.k

        a = -1 / sqrt((k + 1) * (k + 1 + alpha))
        b = -(2 * k + 1 + alpha) / sqrt((k + 1) * (k + 1 + alpha))
        c = sqrt(k * S(k + alpha) / ((k + 1) * (k + 1 + alpha)))

        self.k += 1
        return a, b, c
