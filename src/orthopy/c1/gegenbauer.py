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


class RecurrenceCoefficients:
    def __init__(self, scaling, lmbda, symbolic="auto"):
        self._jacobi_rc = jacobi.RecurrenceCoefficients(
            scaling, lmbda, lmbda, symbolic=symbolic
        )
        self.int_1 = self._jacobi_rc.int_1

    def __getitem__(self, N):
        return self._jacobi_rc[N]
