import itertools

from ..tools import Iterator1D
from . import gegenbauer


def tree(n, *args, **kwargs):
    return list(itertools.islice(Iterator(*args, **kwargs), n + 1))


class Iterator(Iterator1D):
    """Legendre. The first few are:

    standardization == "monic":
        1
        x
        x**2 - 1/3
        x**3 - 3*x/5
        x**4 - 6*x**2/7 + 3/35
        x**5 - 10*x**3/9 + 5*x/21

    standardization == "classical":
        1
        x
        3*x**2/2 - 1/2
        5*x**3/2 - 3*x/2
        35*x**4/8 - 15*x**2/4 + 3/8
        63*x**5/8 - 35*x**3/4 + 15*x/8

    standardization == "normal":
        sqrt(2)/2
        sqrt(6)*x/2
        3*sqrt(10)*x**2/4 - sqrt(10)/4
        5*sqrt(14)*x**3/4 - 3*sqrt(14)*x/4
        105*sqrt(2)*x**4/16 - 45*sqrt(2)*x**2/8 + 9*sqrt(2)/16
        63*sqrt(22)*x**5/16 - 35*sqrt(22)*x**3/8 + 15*sqrt(22)*x/16
    """

    def __init__(self, X, standardization, *args, **kwargs):
        if standardization == "monic":
            iterator = IteratorRCMonic(*args, **kwargs)
        elif standardization == "classical":
            # p(1) = (n+alpha over n)   (=1 if alpha=0)
            iterator = IteratorRCClassical(*args, **kwargs)
        else:
            valid = ", ".join(["monic", "classical", "normal"])
            assert (
                standardization == "normal"
            ), f"Unknown standardization '{standardization}'. (valid: {valid})"
            iterator = IteratorRCNormal(*args, **kwargs)

        super().__init__(X, iterator)


class IteratorRCMonic(gegenbauer.IteratorRCMonic):
    def __init__(self, symbolic=False):
        super().__init__(0, symbolic)


class IteratorRCClassical(gegenbauer.IteratorRCClassical):
    def __init__(self, symbolic=False):
        super().__init__(0, symbolic)


class IteratorRCNormal(gegenbauer.IteratorRCNormal):
    def __init__(self, symbolic=False):
        super().__init__(0, symbolic)
