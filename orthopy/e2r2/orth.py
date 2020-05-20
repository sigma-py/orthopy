import itertools

from ..enr2 import Iterator as IteratorND


def tree(X, n, **kwargs):
    return list(itertools.islice(Iterator(X, **kwargs), n + 1))


class Iterator(IteratorND):
    def __init__(self, X, *args, **kwargs):
        assert X.shape[0] == 2, "X has incorrect shape (X.shape[0] != 2)."
        super().__init__(X, *args, **kwargs)
