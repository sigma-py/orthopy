import itertools

from . import jacobi


def tree(n, *args, **kwargs):
    return list(itertools.islice(Iterator(*args, **kwargs), n + 1))


class Iterator(jacobi.Iterator):
    def __init__(self, X, scaling, lmbda, symbolic):
        super().__init__(X, scaling, lmbda, lmbda, symbolic)
