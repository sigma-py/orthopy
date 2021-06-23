from . import jacobi


def plot(n, scaling, lmbda):
    jacobi.plot(n, scaling, lmbda, lmbda)


def show(*args, **kwargs):
    from matplotlib import pyplot as plt

    plot(*args, **kwargs)
    plt.show()


def savefig(filename, *args, **kwargs):
    from matplotlib import pyplot as plt

    plot(*args, **kwargs)
    plt.savefig(filename, transparent=True, bbox_inches="tight")


class Eval:
    def __init__(self, X, scaling, lmbda, symbolic="auto"):
        self._jacobi_eval = jacobi.Eval(X, scaling, lmbda, lmbda, symbolic=symbolic)

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._jacobi_eval)


class RecurrenceCoefficients(jacobi.RecurrenceCoefficients):
    def __init__(self, scaling, lmbda, symbolic="auto"):
        super().__init__(scaling, lmbda, lmbda, symbolic=symbolic)
