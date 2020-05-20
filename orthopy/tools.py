import math
import sys

import numpy


def math_comb(n, k):
    if sys.version < "3.8":
        if k > n - k:
            k = n - k

        out = 1
        for i in range(k):
            out *= n - i
            out //= i + 1
        return out

    return math.comb(n, k)


class Iterator1D:
    def __init__(self, x, iterator_abc):
        self.iterator_abc = iterator_abc
        self.x = x
        self.k = 0
        self.last = [None, None]

    def __iter__(self):
        return self

    def __next__(self):
        if self.k == 0:
            out = numpy.full(self.x.shape, self.iterator_abc.p0)
        else:
            a, b, c = next(self.iterator_abc)
            out = self.last[0] * (self.x * a - b)
            if self.k > 1:
                out -= self.last[1] * c

        self.last[1] = self.last[0]
        self.last[0] = out
        self.k += 1
        return out


class ProductIterator:
    def __init__(self, rc_iterator, X, symbolic):
        self.rc_iterator = rc_iterator

        self.a = []
        self.b = []
        self.c = []
        dim = X.shape[0]
        self.p0n = rc_iterator.p0 ** dim
        self.k = 0
        self.X = X
        self.last = [None, None]

    def __iter__(self):
        return self

    def __next__(self):
        aa, bb, cc = next(self.rc_iterator)
        a = self.a
        b = self.b
        c = self.c
        a.append(aa)
        b.append(bb)
        c.append(cc)

        X = self.X
        L = self.k
        dim = X.shape[0]

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


def line_tree(t, p0, a, b, c):
    n = len(a)
    assert len(b) == n
    assert len(c) == n

    out = [numpy.ones_like(t) * p0]

    for L in range(n):
        nxt = out[-1] * (t * a[L] - b[L])
        if L > 0:
            nxt -= out[-2] * c[L]
        out.append(nxt)

    return out


def line_evaluate(t, p0, a, b, c):
    """Evaluate the orthogonal polynomial defined by its recurrence coefficients
    a, b, and c at the point(s) t.
    """
    vals1 = numpy.zeros_like(t, dtype=int)
    # The order is important here; see <https://github.com/sympy/sympy/issues/13637>.
    vals2 = numpy.ones_like(t) * p0

    for a_k, b_k, c_k in zip(a, b, c):
        vals0, vals1 = vals1, vals2
        vals2 = vals1 * (t * a_k - b_k) - vals0 * c_k
    return vals2
