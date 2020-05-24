import math
import sys

import numpy


def _math_comb(n, k):
    if sys.version < "3.8":
        if k > n - k:
            k = n - k

        out = 1
        for i in range(k):
            out *= n - i
            out //= i + 1
        return out

    return math.comb(n, k)


def full_like(x, val):
    if isinstance(x, numpy.ndarray):
        return numpy.full_like(x, val)
    # assume x is just a float or int
    return val
