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


def line_tree2(t, iterator, n):
    out = [numpy.ones_like(t) * iterator.p0]

    for L in range(n):
        a, b, c = next(iterator)
        nxt = out[-1] * (t * a - b)
        if L > 0:
            nxt -= out[-2] * c
        out.append(nxt)

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
    # The order is important here; see
    # <https://github.com/sympy/sympy/issues/13637>.
    vals2 = numpy.ones_like(t) * p0

    for a_k, b_k, c_k in zip(a, b, c):
        vals0, vals1 = vals1, vals2
        vals2 = vals1 * (t * a_k - b_k) - vals0 * c_k
    return vals2
