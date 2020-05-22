import itertools

from ..tools import Iterator1D
from . import jacobi


def tree(n, *args, **kwargs):
    return list(itertools.islice(Iterator(*args, **kwargs), n + 1))


class Iterator(Iterator1D):
    def __init__(self, X, scaling, *args, **kwargs):
        cls = {
            "monic": IteratorRCMonic,
            "classical": IteratorRCClassical,
            "normal": IteratorRCNormal,
        }[scaling]
        super().__init__(X, cls(*args, **kwargs))


class IteratorRCMonic(jacobi.IteratorRCMonic):
    def __init__(self, lmbda, symbolic=False):
        super().__init__(lmbda, lmbda, symbolic)


class IteratorRCClassical(jacobi.IteratorRCClassical):
    def __init__(self, lmbda, symbolic=False):
        super().__init__(lmbda, lmbda, symbolic)


class IteratorRCNormal(jacobi.IteratorRCNormal):
    def __init__(self, lmbda, symbolic=False):
        super().__init__(lmbda, lmbda, symbolic)
