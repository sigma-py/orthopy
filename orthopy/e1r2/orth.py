import itertools
import math

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
        rc = {
            "probabilst": RCMonic,
            "monic": RCMonic,
            "physicist": RCPhysicist,
            "normal": RCNormal,
        }[scaling]
        super().__init__(X, rc(*args, **kwargs))


class RCMonic:
    def __init__(self, symbolic=False):
        self.p0 = 1

    def __getitem__(self, k):
        a = 1
        b = 0
        # Note: The first c is never actually used.
        c = k
        return a, b, c


class RCPhysicist:
    def __init__(self, symbolic=False):
        self.p0 = 1

    def __getitem__(self, k):
        a = 2
        b = 0
        # Note: The first c is never actually used.
        c = 2 * k
        return a, b, c


class RCNormal:
    def __init__(self, symbolic=False):
        self.sqrt = sympy.sqrt if symbolic else math.sqrt
        self.frac = sympy.Rational if symbolic else lambda a, b: a / b
        pi = sympy.pi if symbolic else math.pi

        self.p0 = 1 / self.sqrt(self.sqrt(pi))

    def __getitem__(self, k):
        a = self.sqrt(self.frac(2, k + 1))
        b = 0
        # Note: The first c is never actually used.
        c = self.sqrt(self.frac(k, k + 1))
        return a, b, c
