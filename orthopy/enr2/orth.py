import itertools

from ..e1r2.orth import RCNormal
from ..tools import ProductIterator


def tree(X, n, symbolic=False):
    return list(itertools.islice(Iterator(X, symbolic), n + 1))


class Iterator(ProductIterator):
    def __init__(self, X, symbolic=False):
        rc = RCNormal(symbolic)
        super().__init__(rc, X, symbolic)
