import itertools

from ..c1.recurrence_coefficients import Legendre
from ..tools import ProductIterator


def tree(n, *args, **kwargs):
    return list(itertools.islice(Iterator(*args, **kwargs), n + 1))


class Iterator(ProductIterator):
    def __init__(self, X, symbolic=False):
        iterator = Legendre("normal", symbolic)
        super().__init__(iterator, X, symbolic)
