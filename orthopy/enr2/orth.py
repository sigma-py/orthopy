import itertools

import numpy

from ..e1r2 import recurrence_coefficients
from ..tools import math_comb


def tree(X, n, symbolic=False):
    return list(itertools.islice(Iterator(X, n, symbolic), n + 1))


class Iterator:
    def __init__(self, X, n, symbolic=False):
        self.p0, self.a, self.b, self.c = recurrence_coefficients(n + 1, "normal", symbolic=symbolic)
        self.dim = X.shape[0]
        self.p0n = self.p0 ** self.dim
        self.k = 0
        self.n = n
        self.X = X
        self.last = [None, None]

    def __iter__(self):
        return self

    def __next__(self):
        X = self.X
        L = self.k
        dim = self.dim
        a = self.a
        b = self.b
        c = self.c

        if L == 0:
            out = numpy.full([1] + list(X.shape[1:]), self.p0n)
        else:
            level = []
            for i in range(dim - 1):
                m1 = math_comb(L + dim - i - 2, dim - i - 1)
                last0 = self.last[0][-m1:]
                if L > 1:
                    m2 = math_comb(L + dim - i - 3, dim - i - 1)
                    last1 = self.last[1][-m2:]
                r = 0
                for k in range(L):
                    m = math_comb(k + dim - i - 2, dim - i - 2)
                    val = last0[r : r + m] * (a[L - k - 1] * X[i] - b[L - k - 1])
                    if L - k > 1:
                        val -= last1[r : r + m] * c[L - k - 1]
                    r += m
                    level.append(val)

            # treat the last one separately
            val = self.last[0][-1] * (a[L - 1] * X[-1] - b[L - 1])
            if L > 1:
                val -= self.last[1][-1] * c[L - 1]
            level.append([val])

            out = numpy.concatenate(level)

        self.last[1] = self.last[0]
        self.last[0] = out
        self.k += 1
        return out
