import itertools

from . import jacobi


def tree(n, *args, **kwargs):
    return list(itertools.islice(Eval(*args, **kwargs), n + 1))


def plot(n, lmbda, scaling):
    jacobi.plot(n, lmbda, lmbda, scaling)


def show(*args, **kwargs):
    import matplotlib.pyplot as plt

    plot(*args, **kwargs)
    plt.show()


def savefig(filename, *args, **kwargs):
    import matplotlib.pyplot as plt

    plot(*args, **kwargs)
    plt.savefig(filename, transparent=True, bbox_inches="tight")


class Eval(jacobi.Eval):
    def __init__(self, X, lmbda, scaling, symbolic=False):
        super().__init__(X, lmbda, lmbda, scaling, symbolic)


class RecurrenceCoefficients(jacobi.RecurrenceCoefficients):
    def __init__(self, lmbda, scaling, symbolic=False):
        super().__init__(lmbda, lmbda, scaling, symbolic)
