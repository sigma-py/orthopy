import itertools

from ..e1r2.orth import RCPhysicistNormal, RCProbabilistNormal
from ..helpers import ProductIterator


def tree(n, *args, **kwargs):
    return list(itertools.islice(Iterator(*args, **kwargs), n + 1))


class Iterator(ProductIterator):
    def __init__(self, X, standardization, symbolic=False):
        rc = {"probabilist": RCProbabilistNormal, "physicist": RCPhysicistNormal,}[
            standardization
        ](symbolic)
        super().__init__(rc, X, symbolic)
