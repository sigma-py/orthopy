import numpy as np
import sympy

from ..e1r2.main import RCPhysicistNormal, RCProbabilistNormal
from ..helpers import ProductEval, ProductEvalWithDegrees


class Eval(ProductEval):
    def __init__(self, X, standardization, symbolic="auto"):
        if symbolic == "auto":
            symbolic = np.asarray(X).dtype == sympy.Basic

        rc = {"probabilists": RCProbabilistNormal, "physicists": RCPhysicistNormal}[
            standardization
        ](symbolic)

        sqrt = sympy.sqrt if symbolic else np.sqrt
        pi = sympy.pi if symbolic else np.pi
        int_1 = sqrt(pi)
        super().__init__(rc, int_1, X, symbolic)


class EvalWithDegrees(ProductEvalWithDegrees):
    def __init__(self, X, standardization, symbolic="auto"):
        if symbolic == "auto":
            symbolic = np.asarray(X).dtype == sympy.Basic

        rc = {"probabilists": RCProbabilistNormal, "physicists": RCPhysicistNormal}[
            standardization
        ](symbolic)

        sqrt = sympy.sqrt if symbolic else np.sqrt
        pi = sympy.pi if symbolic else np.pi
        int_1 = sqrt(pi)
        super().__init__(rc, int_1, X, symbolic)
