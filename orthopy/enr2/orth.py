import itertools

from ..e1r2.orth import IteratorRCNormal
from ..tools import ProductIterator


def tree(X, n, symbolic=False):
    return list(itertools.islice(Iterator(X, symbolic), n + 1))


class Iterator(ProductIterator):
    def __init__(self, X, symbolic=False):
        iterator = IteratorRCNormal(symbolic)
        super().__init__(iterator, X, symbolic)
