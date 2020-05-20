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
    """Evaluates the entire tree of orthogonal polynomials for the n-cube

    The computation is organized such that tree returns a list of arrays, L={0, ...,
    dim}, where each level corresponds to the polynomial degree L.  Further, each level
    is organized like a discrete (dim-1)-dimensional simplex. Let's demonstrate this for
    3D:

    L = 1:
         (0, 0, 0)

    L = 2:
         (1, 0, 0)
         (0, 1, 0) (0, 0, 1)

    L = 3:
         (2, 0, 0)
         (1, 1, 0) (1, 0, 1)
         (0, 2, 0) (0, 1, 1) (0, 0, 2)

    The main insight here that makes computation for n dimensions easy is that the next
    level is composed by:

       * Taking the whole previous level and adding +1 to the first entry.
       * Taking the last row of the previous level and adding +1 to the second entry.
       * Taking the last entry of the last row of the previous and adding +1 to the
         third entry.

    In the same manner this can be repeated for `dim` dimensions.
    """

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
        a = self.a
        b = self.b
        c = self.c

        X = self.X
        L = self.k
        dim = X.shape[0]

        if L == 0:
            out = numpy.full([1] + list(X.shape[1:]), self.p0n)
        else:
            aa, bb, cc = next(self.rc_iterator)
            a.append(aa)
            b.append(bb)
            c.append(cc)

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
