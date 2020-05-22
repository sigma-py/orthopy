import itertools

import numpy
import sympy

from ..tools import Iterator1D


def tree(n, *args, **kwargs):
    return list(itertools.islice(Iterator(*args, **kwargs), n + 1))


class Iterator(Iterator1D):
    """Recurrence coefficients for Hermite polynomials.

    Check <https://en.wikipedia.org/wiki/Hermite_polynomials> for the different
    scalings.

    The first few are:

    scaling in ["probabilist", "monic"]:
        1
        x
        x**2 - 1
        x**3 - 3*x
        x**4 - 6*x**2 + 3
        x**5 - 10*x**3 + 15*x

    scaling == "physicist":
        1
        2*x
        4*x**2 - 2
        8*x**3 - 12*x
        16*x**4 - 48*x**2 + 12
        32*x**5 - 160*x**3 + 120*x

    scaling == "normal":
        pi**(-1/4)
        sqrt(2)*x/pi**(1/4)
        sqrt(2)*x**2/pi**(1/4) - sqrt(2)/(2*pi**(1/4))
        2*sqrt(3)*x**3/(3*pi**(1/4)) - sqrt(3)*x/pi**(1/4)
        sqrt(6)*x**4/(3*pi**(1/4)) - sqrt(6)*x**2/pi**(1/4) + sqrt(6)/(4*pi**(1/4))
        2*sqrt(15)*x**5/(15*pi**(1/4)) - 2*sqrt(15)*x**3/(3*pi**(1/4)) + sqrt(15)*x/(2*pi**(1/4))
    """

    def __init__(self, X, scaling, *args, **kwargs):
        if scaling in ["probabilist", "monic"]:
            iterator = IteratorRCMonic(*args, **kwargs)
        elif scaling == "physicist":
            iterator = IteratorRCPhysicist(*args, **kwargs)
        else:
            assert scaling == "normal", f"Unknown scaling '{scaling}'."
            iterator = IteratorRCNormal(*args, **kwargs)

        super().__init__(X, iterator)


class IteratorRCMonic:
    def __init__(self, symbolic=False):
        self.k = 0
        self.p0 = 1

    def __iter__(self):
        return self

    def __next__(self):
        a = 1
        b = 0
        # Note: The first c is never actually used.
        c = self.k

        self.k += 1
        return a, b, c


class IteratorRCPhysicist:
    def __init__(self, symbolic=False):
        self.k = 0
        self.p0 = 1

    def __iter__(self):
        return self

    def __next__(self):
        a = 2
        b = 0
        # Note: The first c is never actually used.
        c = 2 * self.k

        self.k += 1
        return a, b, c


class IteratorRCNormal:
    def __init__(self, symbolic=False):
        self.k = 0

        self.sqrt = sympy.sqrt if symbolic else numpy.sqrt
        self.frac = sympy.Rational if symbolic else lambda a, b: a / b
        pi = sympy.pi if symbolic else numpy.pi

        self.p0 = 1 / self.sqrt(self.sqrt(pi))

    def __iter__(self):
        return self

    def __next__(self):
        a = self.sqrt(self.frac(2, self.k + 1))
        b = 0
        # Note: The first c is never actually used.
        c = self.sqrt(self.frac(self.k, self.k + 1))

        self.k += 1
        return a, b, c
