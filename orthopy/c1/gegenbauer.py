import itertools

from . import jacobi


def tree(n, *args, **kwargs):
    return list(itertools.islice(Iterator(*args, **kwargs), n + 1))


class Iterator(jacobi.Iterator):
    def __init__(self, X, lmbda, scaling, symbolic=False):
        super().__init__(X, lmbda, lmbda, scaling, symbolic)


class RecurrenceCoefficients(jacobi.RecurrenceCoefficients):
    def __init__(self, lmbda, scaling, symbolic=False):
        super().__init__(scaling, lmbda, lmbda, scaling, symbolic)
