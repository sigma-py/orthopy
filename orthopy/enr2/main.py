from ..e1r2.main import RCPhysicistNormal, RCProbabilistNormal
from ..helpers import ProductEval


class Eval(ProductEval):
    def __init__(self, X, standardization, symbolic="auto"):
        rc = {"probabilists": RCProbabilistNormal, "physicists": RCPhysicistNormal}[
            standardization
        ](symbolic)
        super().__init__(rc, X, symbolic)
