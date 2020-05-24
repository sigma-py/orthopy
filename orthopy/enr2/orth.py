import itertools

from ..e1r2.orth import RCPhysicistNormal
from ..helpers import ProductIterator


def tree(X, n, symbolic=False):
    return list(itertools.islice(Iterator(X, symbolic), n + 1))


class Iterator(ProductIterator):
    def __init__(self, X, symbolic=False):
        rc = RCPhysicistNormal(symbolic)
        super().__init__(rc, X, symbolic)
