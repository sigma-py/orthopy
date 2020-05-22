import itertools

import numpy

from ..cn import Iterator as CNIterator


def tree(n, *args, **kwargs):
    return list(itertools.islice(Iterator(*args, **kwargs), n + 1))


class Iterator(CNIterator):
    def __init__(self, X, **kwargs):
        X = numpy.asarray(X)
        assert X.shape[0] == 3, "X has incorrect shape (X.shape[0] != 3)."
        super().__init__(X, **kwargs)
