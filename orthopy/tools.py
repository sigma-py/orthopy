# -*- coding: utf-8 -*-
#
import numpy


def line_tree(t, p0, a, b, c):
    n = len(a)
    assert len(b) == n
    assert len(c) == n

    out = [numpy.ones_like(t) * p0]

    for L in range(n):
        out.append(out[L] * (t*a[L] - b[L]))
        if L > 0:
            out[L+1] -= out[L-1] * c[L]

    return out
