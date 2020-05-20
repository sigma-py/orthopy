import numpy

from .tools import math_comb


class ProductIterator:
    def __init__(self, p0, a, b, c, X, symbolic):
        self.a = a
        self.b = b
        self.c = c
        dim = X.shape[0]
        self.p0n = p0 ** dim
        self.k = 0
        self.X = X
        self.last = [None, None]

    def __iter__(self):
        return self

    def __next__(self):
        X = self.X
        L = self.k
        dim = X.shape[0]
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
