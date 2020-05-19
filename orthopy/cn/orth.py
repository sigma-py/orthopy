import itertools

import numpy

from ..c1.recurrence_coefficients import Legendre
from ..tools import math_comb


def tree(X, n, symbolic=False):
    return list(itertools.islice(Iterator(X, n, symbolic), n + 1))


class Iterator:
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

    def __init__(self, X, n, symbolic=False):
        self.legendre_iterator = Legendre("normal", symbolic=symbolic)
        self.p0 = self.legendre_iterator.p0
        self.dim = X.shape[0]
        self.p0n = self.p0 ** self.dim
        self.last = [None, None]
        self.k = 0
        self.X = X
        self.a = []
        self.b = []
        self.c = []

    def __iter__(self):
        return self

    def __next__(self):
        aa, bb, cc = next(self.legendre_iterator)
        dim = self.dim
        L = self.k
        X = self.X
        a = self.a
        b = self.b
        c = self.c
        a.append(aa)
        b.append(bb)
        c.append(cc)

        if L == 0:
            out = numpy.full([1] + list(X.shape[1:]), self.p0n)
        else:
            level = []
            for i in range(dim - 1):
                m1 = math_comb(L + dim - i - 2, dim - i - 1)
                if L > 1:
                    m2 = math_comb(L + dim - i - 3, dim - i - 1)
                r = 0
                for k in range(L):
                    m = math_comb(k + dim - i - 2, dim - i - 2)
                    val = self.last[0][-m1:][r : r + m] * (
                        a[L - k - 1] * X[i] - b[L - k - 1]
                    )
                    if L - k > 1:
                        val -= self.last[1][-m2:][r : r + m] * c[L - k - 1]
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
