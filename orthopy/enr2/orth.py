import itertools

from ..e1r2 import IteratorRC
from ..tools import ProductIterator


def tree(X, n, symbolic=False):
    return list(itertools.islice(Iterator(X, n, symbolic), n + 1))


class Iterator(ProductIterator):
    # TODO remove n argument
    def __init__(self, X, n, symbolic=False):
        iterator = IteratorRC("normal", symbolic)
        p0 = iterator.p0
        a = []
        b = []
        c = []
        for abc in itertools.islice(iterator, n + 1):
            a.append(abc[0])
            b.append(abc[1])
            c.append(abc[2])

        super().__init__(p0, a, b, c, X, symbolic)
