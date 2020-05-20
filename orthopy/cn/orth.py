import itertools

from ..c1 import legendre
from ..tools import ProductIterator


def tree(n, *args, **kwargs):
    return list(itertools.islice(Iterator(*args, **kwargs), n + 1))


class Iterator(ProductIterator):
    def __init__(self, X, symbolic=False):
        iterator = legendre.IteratorRCNormal(symbolic)
        super().__init__(iterator, X, symbolic)
