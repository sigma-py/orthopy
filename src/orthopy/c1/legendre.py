from . import gegenbauer


def plot(n, scaling):
    gegenbauer.plot(n, scaling, 0)


def show(*args, **kwargs):
    from matplotlib import pyplot as plt

    plot(*args, **kwargs)
    plt.show()


def savefig(filename, *args, **kwargs):
    from matplotlib import pyplot as plt

    plot(*args, **kwargs)
    plt.savefig(filename, transparent=True, bbox_inches="tight")


class Eval:
    """Legendre. The first few are:

    scaling == "monic":
        1
        x
        x**2 - 1/3
        x**3 - 3*x/5
        x**4 - 6*x**2/7 + 3/35
        x**5 - 10*x**3/9 + 5*x/21

    scaling == "classical":
        1
        x
        3*x**2/2 - 1/2
        5*x**3/2 - 3*x/2
        35*x**4/8 - 15*x**2/4 + 3/8
        63*x**5/8 - 35*x**3/4 + 15*x/8

    scaling == "normal":
        sqrt(2)/2
        sqrt(6)*x/2
        3*sqrt(10)*x**2/4 - sqrt(10)/4
        5*sqrt(14)*x**3/4 - 3*sqrt(14)*x/4
        105*sqrt(2)*x**4/16 - 45*sqrt(2)*x**2/8 + 9*sqrt(2)/16
        63*sqrt(22)*x**5/16 - 35*sqrt(22)*x**3/8 + 15*sqrt(22)*x/16
    """

    def __init__(self, X, scaling, symbolic="auto"):
        self._gegenbauer_eval = gegenbauer.Eval(X, scaling, 0, symbolic)

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._gegenbauer_eval)


class RecurrenceCoefficients(gegenbauer.RecurrenceCoefficients):
    def __init__(self, scaling, symbolic):
        super().__init__(scaling, 0, symbolic)
