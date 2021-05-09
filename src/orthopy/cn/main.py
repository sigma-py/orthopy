import numpy as np
import sympy

from ..c1 import jacobi
from ..helpers import ProductEval, ProductEvalWithDegrees


class Eval(ProductEval):
    def __init__(self, X, alpha=0, beta=0, symbolic="auto"):
        if symbolic == "auto":
            symbolic = np.asarray(X).dtype == sympy.Basic

        rc = jacobi.RecurrenceCoefficients("normal", alpha, beta, symbolic)
        super().__init__(rc, 1, X, symbolic)


class EvalWithDegrees(ProductEvalWithDegrees):
    def __init__(self, X, alpha=0, beta=0, symbolic="auto"):
        if symbolic == "auto":
            symbolic = np.asarray(X).dtype == sympy.Basic

        rc = jacobi.RecurrenceCoefficients("normal", alpha, beta, symbolic)
        super().__init__(rc, 1, X, symbolic)
