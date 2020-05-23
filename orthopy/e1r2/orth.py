import itertools
import math

import sympy

from ..tools import Iterator1D


def tree(n, *args, **kwargs):
    return list(itertools.islice(Iterator(*args, **kwargs), n + 1))


class Iterator(Iterator1D):
    """Recurrence coefficients for Hermite polynomials.

    The physicicist's standardization is with respect to the weight function exp(-x^2),
    the probabilists' standardization is with respect to the weight function
    1/sqrt(2*pi) * exp(-x^2/2). See <https://en.wikipedia.org/wiki/Hermite_polynomials>.

    The first few are:
    Probabilist:
        scaling in ["monic", "classical"]:
            1
            x
            x**2 - 1
            x**3 - 3*x
            x**4 - 6*x**2 + 3
            x**5 - 10*x**3 + 15*x

        scaling == "normal":
            1
            x
            sqrt(2)*x**2/2 - sqrt(2)/2
            sqrt(6)*x**3/6 - sqrt(6)*x/2
            sqrt(6)*x**4/12 - sqrt(6)*x**2/2 + sqrt(6)/4
            sqrt(30)*x**5/60 - sqrt(30)*x**3/6 + sqrt(30)*x/4

    Physicist:
        scaling == "monic":
            1
            x
            x**2 - 1/2
            x**3 - 3*x/2
            x**4 - 3*x**2 + 3/4
            x**5 - 5*x**3 + 15*x/4

        scaling == "classical":
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

    def __init__(self, X, standardization, scaling, *args, **kwargs):
        rc = {
            "probabilist": {
                # The classical scheme is monic
                "monic": RCProbabilistMonic,
                "classical": RCProbabilistMonic,
                "normal": RCProbabilistNormal,
            },
            "physicist": {
                "monic": RCPhysicistMonic,
                "classical": RCPhysicistClassical,
                "normal": RCPhysicistNormal,
            },
        }[standardization][scaling](*args, **kwargs)
        super().__init__(X, rc)


class RCProbabilistMonic:
    def __init__(self, symbolic=False):
        self.nan = None if symbolic else math.nan
        self.p0 = 1

    def __getitem__(self, k):
        a = 1
        b = 0
        c = k if k > 0 else self.nan
        return a, b, c


class RCProbabilistNormal:
    def __init__(self, symbolic=False):
        self.frac = sympy.Rational if symbolic else lambda a, b: a / b
        self.nan = None if symbolic else math.nan
        self.sqrt = sympy.sqrt if symbolic else math.sqrt
        self.p0 = 1

    def __getitem__(self, k):
        a = 1 / self.sqrt(k + 1)
        b = 0
        c = self.sqrt(self.frac(k, k + 1)) if k > 0 else self.nan
        return a, b, c


class RCPhysicistMonic:
    def __init__(self, symbolic=False):
        self.frac = sympy.Rational if symbolic else lambda a, b: a / b
        self.nan = None if symbolic else math.nan
        self.p0 = 1

    def __getitem__(self, k):
        a = 1
        b = 0
        c = self.frac(k, 2) if k > 0 else self.nan
        return a, b, c


class RCPhysicistClassical:
    def __init__(self, symbolic=False):
        self.nan = None if symbolic else math.nan
        self.p0 = 1

    def __getitem__(self, k):
        a = 2
        b = 0
        c = 2 * k if k > 0 else self.nan
        return a, b, c


class RCPhysicistNormal:
    def __init__(self, symbolic=False):
        self.frac = sympy.Rational if symbolic else lambda a, b: a / b
        self.nan = None if symbolic else math.nan
        self.sqrt = sympy.sqrt if symbolic else math.sqrt

        pi = sympy.pi if symbolic else math.pi
        self.p0 = 1 / self.sqrt(self.sqrt(pi))

    def __getitem__(self, k):
        a = self.sqrt(self.frac(2, k + 1))
        b = 0
        c = self.sqrt(self.frac(k, k + 1)) if k > 0 else self.nan
        return a, b, c
