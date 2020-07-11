import numpy
import sympy

from . import gegenbauer


def plot(n, scaling):
    gegenbauer.plot(n, +0.5, scaling)


def show(*args, **kwargs):
    from matplotlib import pyplot as plt

    plot(*args, **kwargs)
    plt.show()


def savefig(filename, *args, **kwargs):
    from matplotlib import pyplot as plt

    plot(*args, **kwargs)
    plt.savefig(filename, transparent=True, bbox_inches="tight")


class Eval(gegenbauer.Eval):
    """Chebyshev polynomials of the second kind. The first few are:

    scaling == "monic":
        1
        x
        x**2 - 1/4
        x**3 - x/2
        x**4 - 3*x**2/4 + 1/16
        x**5 - x**3 + 3*x/16

    scaling == "classical":
        1
        3*x/2
        5*x**2/2 - 5/8
        35*x**3/8 - 35*x/16
        63*x**4/8 - 189*x**2/32 + 63/128
        231*x**5/16 - 231*x**3/16 + 693*x/256

    scaling == "normal":
        sqrt(2)/sqrt(pi)
        2*sqrt(2)*x/sqrt(pi)
        4*sqrt(2)*x**2/sqrt(pi) - sqrt(2)/sqrt(pi)
        8*sqrt(2)*x**3/sqrt(pi) - 4*sqrt(2)*x/sqrt(pi)
        16*sqrt(2)*x**4/sqrt(pi) - 12*sqrt(2)*x**2/sqrt(pi) + sqrt(2)/sqrt(pi)
        32*sqrt(2)*x**5/sqrt(pi) - 32*sqrt(2)*x**3/sqrt(pi) + 6*sqrt(2)*x/sqrt(pi)
    """

    def __init__(self, X, scaling, symbolic="auto"):
        if symbolic == "auto":
            symbolic = numpy.asarray(X).dtype == sympy.Basic

        lmbda = sympy.S(1) / 2 if symbolic else 0.5
        super().__init__(X, scaling, lmbda, symbolic)
