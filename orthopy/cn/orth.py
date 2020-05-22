import itertools

from ..c1 import jacobi
from ..tools import ProductIterator


def tree(n, *args, **kwargs):
    return list(itertools.islice(Iterator(*args, **kwargs), n + 1))


class Iterator(ProductIterator):
    def __init__(self, X, alpha=0, beta=0, symbolic=False):
        rc = jacobi.RCNormal(alpha, beta, symbolic)
        super().__init__(rc, X, symbolic)
