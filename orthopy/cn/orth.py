import itertools

from ..c1.recurrence_coefficients import Legendre
from ..tools import ProductIterator


def tree(X, n, symbolic=False):
    return list(itertools.islice(Iterator(X, symbolic), n + 1))


class Iterator(ProductIterator):
    def __init__(self, X, symbolic=False):
        iterator = Legendre("normal", symbolic)
        super().__init__(iterator, X, symbolic)
