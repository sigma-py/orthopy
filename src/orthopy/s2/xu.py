from __future__ import annotations

import numpy as np
import sympy
from numpy.typing import ArrayLike

try:
    # Python 3.8+
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


def savefig_single(filename, *args, **kwargs):
    from matplotlib import pyplot as plt

    plot_single(*args, **kwargs)
    plt.savefig(filename, transparent=True, bbox_inches="tight")


def show_single(*args, **kwargs):
    from matplotlib import pyplot as plt

    plot_single(*args, **kwargs)
    plt.show()


def plot_single(*args, **kwargs):
    from .tools import plot_single as ps

    ps("Xu", Eval, *args, **kwargs)


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

    pt("Xu", Eval, *args, **kwargs)


class Eval:
    """Evaluates the entire tree of orthogonal polynomials on the unit disk.

    The return value is a list of arrays, where `out[k]` hosts the `2*k+1`
    values of the `k`th level of the tree

        (0, 0)
        (0, 1)   (1, 1)
        (0, 2)   (1, 2)   (2, 2)
          ...      ...      ...

    See

    Yuan Xu
    Orthogonal polynomials of several variables,
    Jan. 2017,
    <https://arxiv.org/abs/1701.02709>

    equation (3.4) for a formulation in terms of Gegenbauer polynomials C. The
    recurrence relation can be worked out from there.
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
        self.one_min_x2 = 1 - self.X[0] ** 2
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
            alpha, beta, gamma, delta = self.rc[self.L]
            out = np.concatenate(
                [
                    self.last[0] * np.multiply.outer(alpha, self.X[0]),
                    [self.last[0][-1] * beta * self.X[1]],
                ]
            )

            if self.L > 1:
                out[:-2] -= (self.last[1].T * gamma).T
                out[-1] -= self.last[1][-1] * delta * self.one_min_x2

        self.last[1] = self.last[0]
        self.last[0] = out
        self.L += 1
        return out


class RCClassical:
    """The maximum values (which are attained at (1, 0) for the first and (0, 1) for the
    last polynomial in each level) is 1.
    """

    def __init__(self, symbolic: bool, mu: int | float = 1):
        self.S = sympy.S if symbolic else lambda x: x
        self.sqrt = np.vectorize(sympy.sqrt) if symbolic else np.sqrt
        self.mu = mu
        assert self.mu > -1

        self.p0 = 1

    # How to get from the monic representation to this one:
    # The value at (0, 1) needs to be 1, so
    #
    # self.last[0][-1] * beta * self.X[1] - self.last[1][-1] * delta * self.one_min_x2
    # = beta - delta
    #
    # (from the recurrence) needs to be 1. Hence
    #
    #  beta_new = beta / (beta - delta)
    #  delta_new = delta / (beta - delta) / beta_min_delta_from_previous_step
    #
    # Calling z = beta - delta, this establishes the formula
    #
    #     z(n) = beta(n) - delta(n) / z(n-1)
    #
    # which is a continued fraction. Using sympy, it can be shown that
    #
    #   z(n) = (mu + n - 1) / (mu + 2 * (n - 1))
    #
    # so
    #
    #   beta(n) = (mu + 2 * (n - 1)) / (mu + n - 1)
    #
    # and delta accordingly. The same reasoning can be applied to alpha and gamma, the
    # result will be
    #
    #   z(n) = (mu + n) / (mu + 2 * n - 1)
    #
    # TODO only the leftmost and rightmost values are correct
    def __getitem__(self, n: int):
        assert n > 0

        n = self.S(n)
        mu = self.mu

        alpha = np.full(n, (mu + 2 * n - 1) / (mu + n))
        # case distinction for mu==0
        if n == 1:
            beta = 1
        else:
            beta = (mu + 2 * n - 2) / (mu + n - 1)

        if n == 1:
            gamma = None
            delta = None
        else:  # n > 1
            k = np.arange(n - 1)
            gamma = (n - 1 - k) * (n + k + mu - 1) / ((mu + n) * (mu + n - 1))
            delta = (n - 1) / (mu + n - 1)

        return alpha, beta, gamma, delta


class RCMonic:
    """alpha and beta both equal 1."""

    def __init__(self, symbolic: bool, mu: int | float = 1):
        self.S = sympy.S if symbolic else lambda x: x
        self.sqrt = np.vectorize(sympy.sqrt) if symbolic else np.sqrt
        self.mu = mu
        assert self.mu > -1

        self.p0 = 1

    def __getitem__(self, n: int):
        assert n > 0

        n = self.S(n)
        mu = self.mu

        alpha = np.ones(n, dtype=int)
        beta = 1

        if n == 1:
            gamma = None
            delta = None
        else:
            k = np.arange(n - 1)
            gamma = (
                (n - 1 - k) * (n + k + mu - 1) / ((2 * n + mu - 3) * (2 * n + mu - 1))
            )
            # case distinction to avoid undefined expression for mu = 0.
            if n == 2:
                delta = self.S(1) / (mu + 2)
            else:
                delta = (n - 1) * (n + mu - 2) / ((2 * n + mu - 2) * (2 * n + mu - 4))

        return alpha, beta, gamma, delta


class RCNormal:
    """Recurrence coefficients for the disk with the Gegenbauer-style weight function
    $$
       W_{\\mu}(x, y) = \\frac{\\mu + 1}{2\\pi} (1 - x^2 - y^2)^{(\\mu - 1)/2}
    $$
    scaled for normality. The default choice is \\mu=1, giving W_1(x, y) = 1.

    In some other tests, one finds a rescaled mu.
    $$
    \\frac{\\mu + 1/2}{\\pi} (1 - x^2 - y^2)^{\\mu - 1/2}
    $$

    This is based on the first orthonormal basis in section 3.2 of

    Yuan Xu,
    Orthogonal polynomials of several variables,
    <https://arxiv.org/pdf/1701.02709.pdf>,

    specifically equation (3.4)

      P_k^n(x, y) = h_{k,}^{-1}
                    C_{n-k}^{k+\\mu+1/2}(x)
                    \\sqrt{1-x^2}^k C_k^{\\mu}(y/\\sqrt(1-x^2)

    with C_n^{\\lambda} being the Gegenbauer polynomial, scaled such that
    $C_n^{\\lambda}(1)=\\frac{(\\lambda+1)_n}{n!}$ and

      h_{k,n}^2 = \\frac{(2k+2\\mu+1)_{n-k} (2\\mu)_k (\\mu)_k (\\mu+1/2)}
                        {(n-k)!k! (\\mu+1/2)_k (n+\\mu+1/2)}.

    The recurrence coefficients are retrieved by exploiting the Gegenbauer recurrence

       C_n^{\\lambda}(t) =
           1/n (
               + 2t (n+\\lambda+1) C_{n-1}^{\\lambda}(t)
               - (n+2\\lambda-2) C_{n-2}^{\\lambda}(t)
               )

    One gets

        P_k^n(x, y) = + 2 \\alpha_{k,n} x P_k^{n-1}(x, y)
                      - \\beta_{k, n} P_k^{n-2}(x, y)

    with

        \\alpha_{k, n}^2 = \\frac{(n+\\mu+1/2)(n+\\mu-1/2)}{(n-k)(n+k+2\\mu)},
        \\beta_{k, n}^2 = \\frac{(n-k-1) (n+\\mu+1/2)(n+k+2\\mu-1)}
                                {(n-k)(n+\\mu-3/2)(n+k+2\\mu)},

    and

        P_n^n(x, y) = + 2 \\gamma_{k,n} y P_{n-1}^{n-1}(x, y)
                      - \\delta_{k, n} (1-x^2) P_{n-1}^{n-2}(x, y)

    with

        \\gamma_{k, n}^2 = \\frac{(n+\\mu-1)(n+\\mu+1/2)}{n (n+2\\mu-1)},
        \\delta_{k, n}^2 = \\frac{(n+2\\mu-2) (n-1) (n+\\mu-1/2) (n+\\mu+1/2)}
                                 {n (n+2\\mu-1) (n+\\mu-1) (n+\\mu-2)}.
    """

    # default: weight function 1
    def __init__(self, symbolic, mu=1):
        self.S = sympy.S if symbolic else lambda x: x
        self.sqrt = np.vectorize(sympy.sqrt) if symbolic else np.sqrt
        self.mu = mu
        assert self.mu > -1

        pi = sympy.pi if symbolic else np.pi
        self.p0 = 1 / self.sqrt(pi)

    def __getitem__(self, n: int):
        assert n > 0

        n = self.S(n)
        mu = self.mu

        k = np.arange(n)
        alpha = self.sqrt(
            (2 * n + mu + 1) * (2 * n + mu - 1) / ((n - k) * (n + k + mu))
        )

        beta = self.sqrt((2 * n + mu - 2) * (2 * n + mu + 1) / (n * (n + mu - 1)))

        if n == 1:
            gamma = None
            delta = None
        else:
            k = np.arange(n - 1)
            gamma = self.sqrt(
                (n - 1 - k)
                * (2 * n + mu + 1)
                * (n + k + mu - 1)
                / ((n - k) * (2 * n + mu - 3) * (n + k + mu))
            )
            # case distinction to avoid undefined expression for mu = 0.
            if n == 2:
                two = self.S(2)
                delta = self.sqrt((3 + mu) * (5 + mu) / (two * (1 + mu) * (2 + mu)))
            else:
                delta = self.sqrt(
                    (n - 1)
                    * (n + mu - 2)
                    * (2 * n + mu - 1)
                    * (2 * n + mu + 1)
                    / (n * (n + mu - 1) * (2 * n + mu - 2) * (2 * n + mu - 4))
                )
        return alpha, beta, gamma, delta
