import itertools

from ..e1r2.orth import RCPhysicistNormal, RCProbabilistNormal
from ..helpers import ProductEval


def tree(n, *args, **kwargs):
    return list(itertools.islice(Eval(*args, **kwargs), n + 1))


class Eval(ProductEval):
    def __init__(self, X, standardization, symbolic=False):
        rc = {"probabilist": RCProbabilistNormal, "physicist": RCPhysicistNormal}[
            standardization
        ](symbolic)
        super().__init__(rc, X, symbolic)
