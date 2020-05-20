import itertools

import numpy
import sympy

from ..tools import Iterator1D


def tree(X, n, *args, **kwargs):
    return list(itertools.islice(Iterator(X, *args, **kwargs), n + 1))


class Iterator(Iterator1D):
    """Recurrence coefficients for Hermite polynomials.

    The first few are:

    standardization in ["probabilist", "monic"]:
        1
        x
        x**2 - 1
        x**3 - 3*x
        x**4 - 6*x**2 + 3
        x**5 - 10*x**3 + 15*x

    standardization == "physicist":
        1
        2*x
        4*x**2 - 2
        8*x**3 - 12*x
        16*x**4 - 48*x**2 + 12
        32*x**5 - 160*x**3 + 120*x

    standardization == "normal":
        pi**(-1/4)
        sqrt(2)*x/pi**(1/4)
        sqrt(2)*x**2/pi**(1/4) - sqrt(2)/(2*pi**(1/4))
        2*sqrt(3)*x**3/(3*pi**(1/4)) - sqrt(3)*x/pi**(1/4)
        sqrt(6)*x**4/(3*pi**(1/4)) - sqrt(6)*x**2/pi**(1/4) + sqrt(6)/(4*pi**(1/4))
        2*sqrt(15)*x**5/(15*pi**(1/4)) - 2*sqrt(15)*x**3/(3*pi**(1/4)) + sqrt(15)*x/(2*pi**(1/4))
    """
    def __init__(self, X, *args, **kwargs):
        super().__init__(X, IteratorRC(*args, **kwargs))


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
