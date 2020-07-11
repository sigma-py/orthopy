import numpy
import sympy

from ..c1 import jacobi
from ..helpers import ProductEval


class Eval(ProductEval):
    def __init__(self, X, alpha=0, beta=0, symbolic="auto"):
        if symbolic == "auto":
            symbolic = numpy.asarray(X).dtype == sympy.Basic

        rc = jacobi.RecurrenceCoefficients("normal", alpha, beta, symbolic)
        super().__init__(rc, X, symbolic)
