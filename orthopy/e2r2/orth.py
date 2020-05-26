import itertools

from ..enr2 import Iterator as IteratorND


def tree(n, *args, **kwargs):
    return list(itertools.islice(Iterator(*args, **kwargs), n + 1))


class Iterator(IteratorND):
    def __init__(self, X, *args, **kwargs):
        assert len(X) == 2, "X has incorrect shape (len(X) != 2)."
        super().__init__(X, *args, **kwargs)
