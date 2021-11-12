from __future__ import annotations

try:
    # Python 3.8+
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import math

import numpy as np
import sympy
from numpy.typing import ArrayLike

from .tools import plot_single as ps


def savefig_single(filename, *args, **kwargs):
    from matplotlib import pyplot as plt

    plot_single(*args, **kwargs)
    plt.savefig(filename, transparent=True, bbox_inches="tight")


def show_single(*args, **kwargs):
    from matplotlib import pyplot as plt

    plot_single(*args, **kwargs)
    plt.show()


def plot_single(*args, **kwargs):
    ps("Zernike-2", Eval, *args, **kwargs)


def savefig_tree(filename, *args, **kwargs):
    from matplotlib import pyplot as plt

    plot_tree(*args, **kwargs)
    plt.savefig(filename, transparent=True, bbox_inches="tight")


def show_tree(*args, **kwargs):
    from matplotlib import pyplot as plt

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

    def __init__(
        self,
        X: ArrayLike,
        scaling: Literal["classical"] | Literal["monic"] | Literal["normal"],
        symbolic: Literal["auto"] | bool = "auto",
    ):
        if symbolic == "auto":
            symbolic = np.asarray(X).dtype == sympy.Basic

        self.rc = {"classical": RCClassical, "monic": RCMonic, "normal": RCNormal}[
            scaling
        ](symbolic)
        self.X = X
        self.L = 0
        self.last = [None, None]

        pi = sympy.pi if symbolic else np.pi
        self.int_p0 = self.rc.p0 * pi

    def __iter__(self):
        return self

    def __next__(self):
        if self.L == 0:
            out = np.array([0 * self.X[0] + self.rc.p0])
        else:
            alpha, beta, gamma = self.rc[self.L]

            shape = list(self.last[0].shape)
            shape[0] += 1
            out = np.zeros(shape, dtype=self.last[0].dtype)

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
    def __init__(self, _):
        self.p0 = 1

    def __getitem__(self, n: int):
        assert n > 0

        alpha = 1
        beta = 1
        if n == 1:
            gamma = None
        else:
            gamma = 1
        return alpha, beta, gamma


class RCMonic:
    def __init__(self, symbolic: bool):
        self.S = sympy.S if symbolic else lambda x: x
        self.p0 = 1

    def __getitem__(self, n: int):
        assert n > 0

        n = self.S(n)
        alpha = np.array([i / n for i in range(1, n + 1)])
        beta = np.array([(n - i) / n for i in range(n)])
        if n == 1:
            gamma = None
        else:
            gamma = np.array([i * (n - i) / (n * (n - 1)) for i in range(1, n)])
        return alpha, beta, gamma


class RCNormal:
    def __init__(self, symbolic: bool):
        self.S = sympy.S if symbolic else lambda x: x
        self.sqrt = sympy.sqrt if symbolic else math.sqrt
        pi = sympy.pi if symbolic else math.pi
        self.p0 = 1 / self.sqrt(pi)

    def __getitem__(self, n: int):
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
