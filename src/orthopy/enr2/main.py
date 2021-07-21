import numpy as np
import sympy

from ..e1r2.main import RCPhysicistNormal, RCProbabilistNormal
from ..helpers import ProductEval, ProductEvalWithDegrees


class Eval:
    def __init__(self, X, standardization, symbolic="auto", return_degrees=False):
        if symbolic == "auto":
            symbolic = np.asarray(X).dtype == sympy.Basic

        rc = {"probabilists": RCProbabilistNormal, "physicists": RCPhysicistNormal}[
            standardization
        ](symbolic)

        sqrt = sympy.sqrt if symbolic else np.sqrt
        pi = sympy.pi if symbolic else np.pi
        int_1 = sqrt(pi) if standardization == "physicists" else 1

        cls = ProductEvalWithDegrees if return_degrees else ProductEval
        self._product_eval = cls(rc, int_1, X)
        self.int_p0 = self._product_eval.int_p0

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._product_eval)
