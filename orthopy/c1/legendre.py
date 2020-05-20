import itertools

from ..tools import Iterator1D
from . import gegenbauer


def tree(n, *args, **kwargs):
    return list(itertools.islice(Iterator(*args, **kwargs), n + 1))


def recurrence_coefficients(n, *args, **kwargs):
    return list(itertools.islice(IteratorRC(*args, **kwargs), n + 1))


class Iterator(Iterator1D):
    def __init__(self, X, *args, **kwargs):
        super().__init__(X, IteratorRC(*args, **kwargs))


class IteratorRC(gegenbauer.IteratorRC):
    def __init__(self, *args, **kwargs):
        super().__init__(0, *args, **kwargs)
