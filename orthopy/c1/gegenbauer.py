import itertools

from ..tools import Iterator1D
from . import jacobi


def tree(n, *args, **kwargs):
    return list(itertools.islice(Iterator(*args, **kwargs), n + 1))


class Iterator(Iterator1D):
    def __init__(self, X, scaling, *args, **kwargs):
        cls = {"monic": RCMonic, "classical": RCClassical, "normal": RCNormal}[scaling]
        super().__init__(X, cls(*args, **kwargs))


class RCMonic(jacobi.RCMonic):
    def __init__(self, lmbda, symbolic=False):
        super().__init__(lmbda, lmbda, symbolic)


class RCClassical(jacobi.RCClassical):
    def __init__(self, lmbda, symbolic=False):
        super().__init__(lmbda, lmbda, symbolic)


class RCNormal(jacobi.RCNormal):
    def __init__(self, lmbda, symbolic=False):
        super().__init__(lmbda, lmbda, symbolic)
