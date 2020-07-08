from . import jacobi


def plot(n, lmbda, scaling):
    jacobi.plot(n, lmbda, lmbda, scaling)


def show(*args, **kwargs):
    from matplotlib import pyplot as plt

    plot(*args, **kwargs)
    plt.show()


def savefig(filename, *args, **kwargs):
    from matplotlib import pyplot as plt

    plot(*args, **kwargs)
    plt.savefig(filename, transparent=True, bbox_inches="tight")


class Eval(jacobi.Eval):
    def __init__(self, X, lmbda, scaling, symbolic="auto"):
        super().__init__(X, lmbda, lmbda, scaling, symbolic=symbolic)


class RecurrenceCoefficients(jacobi.RecurrenceCoefficients):
    def __init__(self, lmbda, scaling, symbolic="auto"):
        super().__init__(lmbda, lmbda, scaling, symbolic=symbolic)
