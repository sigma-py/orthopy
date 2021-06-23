import numpy as np
import sympy

from . import gegenbauer


def plot(n, scaling):
    gegenbauer.plot(n, scaling, -0.5)


def show(*args, **kwargs):
    from matplotlib import pyplot as plt

    plot(*args, **kwargs)
    plt.show()


def savefig(filename, *args, **kwargs):
    from matplotlib import pyplot as plt

    plot(*args, **kwargs)
    plt.savefig(filename, transparent=True, bbox_inches="tight")


class Eval:
    """Chebyshev polynomials of the first kind. The first few are:

    scaling == "monic":
        1
        x
        x**2 - 1/2
        x**3 - 3*x/4
        x**4 - x**2 + 1/8
        x**5 - 5*x**3/4 + 5*x/16

    scaling == "classical":
        1
        x/2
        3*x**2/4 - 3/8
        5*x**3/4 - 15*x/16
        35*x**4/16 - 35*x**2/16 + 35/128
        63*x**5/16 - 315*x**3/64 + 315*x/256

    scaling == "normal":
        1/sqrt(pi)
        sqrt(2)*x/sqrt(pi)
        2*sqrt(2)*x**2/sqrt(pi) - sqrt(2)/sqrt(pi)
        4*sqrt(2)*x**3/sqrt(pi) - 3*sqrt(2)*x/sqrt(pi)
        8*sqrt(2)*x**4/sqrt(pi) - 8*sqrt(2)*x**2/sqrt(pi) + sqrt(2)/sqrt(pi)
        16*sqrt(2)*x**5/sqrt(pi) - 20*sqrt(2)*x**3/sqrt(pi) + 5*sqrt(2)*x/sqrt(pi)

    Another common scaling is via the recurrence
       1
       x
       2 * x * Tn(x) - T{n-1}(x)
    which leads to the equivalence

       Tn(x) = cos(n arccos(x)).

    This isn't implemented here. It can be retrieved from monic scaling by multiplying
    with 2 ** {n-1}. Perhaps this scaling should be added?
    """

    def __init__(self, X, scaling: str, symbolic="auto"):
        if symbolic == "auto":
            symbolic = np.asarray(X).dtype == sympy.Basic

        lmbda = -sympy.S(1) / 2 if symbolic else -0.5
        self._gegenbauer_eval = gegenbauer.Eval(X, scaling, lmbda, symbolic)

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._gegenbauer_eval)
