from __future__ import annotations

import math

try:
    # Python 3.8+
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import numpy as np
import sympy

from ..helpers import Eval1D


class Eval:
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
        -x + 1
        x**2/2 - 2*x + 1
        -x**3/6 + 3*x**2/2 - 3*x + 1
        x**4/24 - 2*x**3/3 + 3*x**2 - 4*x + 1
        -x**5/120 + 5*x**4/24 - 5*x**3/3 + 5*x**2 - 5*x + 1

    The classical and normal standarizations differ for alpha != 0.
    """

    def __init__(self, X, *args, symbolic: Literal["auto"] | bool = "auto", **kwargs):
        if symbolic == "auto":
            symbolic = np.asarray(X).dtype == sympy.Basic

        assert isinstance(symbolic, bool)
        rc = RecurrenceCoefficients(*args, symbolic=symbolic, **kwargs)
        self.int_p0 = rc.p0
        self._eval_1d = Eval1D(X, rc)

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._eval_1d)


class RecurrenceCoefficients:
    def __init__(
        self,
        scaling: Literal["monic"] | Literal["classical"] | Literal["normal"],
        alpha: int | float = 0,
        symbolic: bool = False,
    ):
        cls = {"monic": RCMonic, "classical": RCClassical, "normal": RCNormal}[scaling]
        self.rc = cls(alpha, symbolic)
        self.p0 = self.rc.p0

        gamma = sympy.gamma if symbolic else lambda x: math.gamma(float(x))
        self.int_1 = gamma(alpha + 1)

    def __getitem__(self, N: int):
        return self.rc[N]


class RCMonic:
    def __init__(self, alpha: int | float, symbolic: bool):
        self.nan = None if symbolic else math.nan
        self.alpha = alpha
        self.p0 = 1

    def __getitem__(self, k: int):
        a = 1
        b = 2 * k + 1 + self.alpha
        c = k * (k + self.alpha) if k > 0 else self.nan
        return a, b, c


class RCClassical:
    def __init__(self, alpha: int | float, symbolic: bool):
        self.nan = None if symbolic else math.nan
        self.S = sympy.S if symbolic else lambda a: a
        self.alpha = alpha
        self.p0 = 1

    def __getitem__(self, k):
        alpha = self.alpha
        S = self.S

        a = -S(1) / (k + 1)
        b = -S(2 * k + 1 + alpha) / (k + 1)
        c = S(k + alpha) / (k + 1) if k > 0 else self.nan
        return a, b, c


class RCNormal:
    def __init__(self, alpha: int | float, symbolic: bool):
        self.nan = None if symbolic else math.nan
        self.sqrt = sympy.sqrt if symbolic else math.sqrt
        self.S = sympy.S if symbolic else lambda a: a
        self.alpha = alpha

        gamma = sympy.gamma if symbolic else math.gamma
        self.p0 = 1 / self.sqrt(gamma(alpha + 1))

    def __getitem__(self, k):
        sqrt = self.sqrt
        S = self.S
        alpha = self.alpha

        a = -1 / sqrt((k + 1) * (k + 1 + alpha))
        b = -(2 * k + 1 + alpha) / sqrt((k + 1) * (k + 1 + alpha))
        c = sqrt(k * S(k + alpha) / ((k + 1) * (k + 1 + alpha))) if k > 0 else self.nan
        return a, b, c
