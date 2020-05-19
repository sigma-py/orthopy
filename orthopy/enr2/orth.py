import itertools

from ..e1r2 import recurrence_coefficients
from ..helpers import NDIterator


def tree(X, n, symbolic=False):
    return list(itertools.islice(Iterator(X, n, symbolic), n + 1))


class Iterator(NDIterator):
    def __init__(self, X, n, symbolic=False):
        # TODO remove n argument
        p0, a, b, c = recurrence_coefficients(n + 1, "normal", symbolic=symbolic)
        super().__init__(p0, a, b, c, X, symbolic)
