from ..c1 import jacobi
from ..helpers import ProductEval


class Eval(ProductEval):
    def __init__(self, X, alpha=0, beta=0, symbolic=False):
        rc = jacobi.RCNormal(alpha, beta, symbolic)
        super().__init__(rc, X, symbolic)
