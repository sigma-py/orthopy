import itertools
import numpy
import sympy

from ..tools import Iterator1D


def tree(X, n, standardization, symbolic=False):
    iterator_abc = Iterator(standardization, symbolic)
    return list(itertools.islice(Iterator1D(X, iterator_abc), n + 1))


class Iterator:
    def __init__(self, standardization, symbolic=False):
        self.standardization = standardization
        self.symbolic = symbolic
        self.k = 0

        self.sqrt = sympy.sqrt if symbolic else numpy.sqrt
        self.pi = sympy.pi if symbolic else numpy.pi
        self.frac = sympy.Rational if self.symbolic else lambda a, b: a / b

        if standardization in ["probabilist", "monic"]:
            self.p0 = 1
        elif standardization == "physicist":
            self.p0 = 1
        else:
            assert standardization == "normal", "Unknown standardization '{}'.".format(
                standardization
            )
            self.p0 = 1 / self.sqrt(self.sqrt(self.pi))

    def __iter__(self):
        return self

    def __next__(self):
        frac = self.frac
        pi = self.pi
        sqrt = self.sqrt

        if self.standardization in ["probabilist", "monic"]:
            a = 1
            b = 0
            if self.k == 0:
                # TODO remove
                c = sqrt(pi)  # only used for custom scheme
            else:
                c = self.k
        elif self.standardization == "physicist":
            a = 2
            b = 0
            if self.k == 0:
                # TODO remove
                c = sqrt(pi)  # only used for custom scheme
            else:
                c = 2 * self.k
        else:
            assert self.standardization == "normal"
            a = sqrt(frac(2, self.k + 1))
            b = 0
            if self.k == 0:
                # TODO remove
                c = numpy.nan
            else:
                c = sqrt(frac(self.k, self.k + 1))

        self.k += 1
        return a, b, c


# TODO remove
def recurrence_coefficients(n, standardization, symbolic=False):
    S = numpy.vectorize(sympy.S) if symbolic else lambda x: x
    sqrt = numpy.vectorize(sympy.sqrt) if symbolic else numpy.sqrt
    # TODO replace by sqrt once <https://github.com/numpy/numpy/issues/10363> is fixed
    sS = sympy.S if symbolic else lambda x: x
    ssqrt = sympy.sqrt if symbolic else numpy.sqrt
    pi = sympy.pi if symbolic else numpy.pi

    # Check <https://en.wikipedia.org/wiki/Hermite_polynomials> for the different
    # standardizations.
    N = numpy.array([sS(k) for k in range(n)])
    if standardization in ["probabilist", "monic"]:
        p0 = 1
        a = numpy.ones(n, dtype=int)
        b = numpy.zeros(n, dtype=int)
        c = N
        c[0] = ssqrt(pi)  # only used for custom scheme
    elif standardization == "physicist":
        p0 = 1
        a = numpy.full(n, 2, dtype=int)
        b = numpy.zeros(n, dtype=int)
        c = 2 * N
        c[0] = ssqrt(pi)  # only used for custom scheme
    else:
        assert standardization == "normal", "Unknown standardization '{}'.".format(
            standardization
        )
        p0 = 1 / sqrt(sqrt(pi))
        a = sqrt(S(2) / (N + 1))
        b = numpy.zeros(n, dtype=int)
        c = sqrt(S(N) / (N + 1))
        c[0] = numpy.nan

    return p0, a, b, c
