import itertools

from ..tools import Iterator1D
from . import jacobi


def tree(n, *args, **kwargs):
    return list(itertools.islice(Iterator(*args, **kwargs), n + 1))


class Iterator(Iterator1D):
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


class IteratorRCMonic(jacobi.IteratorRCMonic):
    def __init__(self, lmbda, symbolic=False):
        super().__init__(lmbda, lmbda, symbolic)


class IteratorRCClassical(jacobi.IteratorRCClassical):
    def __init__(self, lmbda, symbolic=False):
        super().__init__(lmbda, lmbda, symbolic)


class IteratorRCNormal(jacobi.IteratorRCNormal):
    def __init__(self, lmbda, symbolic=False):
        super().__init__(lmbda, lmbda, symbolic)
