import itertools

import numpy
import sympy

from ..tools import Iterator1D


def tree(X, n, *args, **kwargs):
    return list(itertools.islice(Iterator(X, *args, **kwargs), n + 1))


class Iterator:
    def __init__(self, X, *args, **kwargs):
        self.iterator1d = Iterator1D(X, IteratorRC(*args, **kwargs))

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.iterator1d)


class IteratorRC:
    def __init__(self, standardization, symbolic=False):
        self.standardization = standardization
        self.symbolic = symbolic
        self.k = 0

        self.sqrt = sympy.sqrt if symbolic else numpy.sqrt
        self.pi = sympy.pi if symbolic else numpy.pi
        self.frac = sympy.Rational if self.symbolic else lambda a, b: a / b

        if standardization in ["probabilist", "monic"]:
            self.p0 = 1
        elif standardization == "physicist":
            self.p0 = 1
        else:
            assert standardization == "normal", "Unknown standardization '{}'.".format(
                standardization
            )
            self.p0 = 1 / self.sqrt(self.sqrt(self.pi))

    def __iter__(self):
        return self

    def __next__(self):
        frac = self.frac
        pi = self.pi
        sqrt = self.sqrt

        # Check <https://en.wikipedia.org/wiki/Hermite_polynomials> for the different
        # standardizations.
        if self.standardization in ["probabilist", "monic"]:
            a = 1
            b = 0
            if self.k == 0:
                # TODO remove
                c = sqrt(pi)  # only used for custom scheme
            else:
                c = self.k
        elif self.standardization == "physicist":
            a = 2
            b = 0
            if self.k == 0:
                # TODO remove
                c = sqrt(pi)  # only used for custom scheme
            else:
                c = 2 * self.k
        else:
            assert self.standardization == "normal"
            a = sqrt(frac(2, self.k + 1))
            b = 0
            if self.k == 0:
                # TODO remove
                c = numpy.nan
            else:
                c = sqrt(frac(self.k, self.k + 1))

        self.k += 1
        return a, b, c
