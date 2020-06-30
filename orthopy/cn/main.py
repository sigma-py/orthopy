import itertools

from ..c1 import jacobi
from ..helpers import ProductEval


def tree(n, *args, **kwargs):
    return list(itertools.islice(Eval(*args, **kwargs), n + 1))


class Eval(ProductEval):
    def __init__(self, X, alpha=0, beta=0, symbolic=False):
        rc = jacobi.RCNormal(alpha, beta, symbolic)
        super().__init__(rc, X, symbolic)
