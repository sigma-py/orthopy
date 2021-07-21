import numpy as np
import sympy

from ..c1 import jacobi
from ..helpers import ProductEval, ProductEvalWithDegrees


class Eval:
    def __init__(self, X, alpha=0, beta=0, symbolic="auto", return_degrees=False):
        if symbolic == "auto":
            symbolic = np.asarray(X).dtype == sympy.Basic

        rc = jacobi.RecurrenceCoefficients("normal", alpha, beta, symbolic)
        cls = ProductEvalWithDegrees if return_degrees else ProductEval
        self._product_eval = cls(rc, 1, X)
        self.int_p0 = self._product_eval.int_p0

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._product_eval)
