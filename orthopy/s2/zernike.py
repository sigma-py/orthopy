import itertools
import math

import numpy
import sympy


def tree(n, *args, **kwargs):
    return list(itertools.islice(Eval(*args, **kwargs), n + 1))


class Eval:
    """
    Torben B. Andersen,
    Efficient and robust recurrence relations for the Zernike circle polynomials and
    their derivatives in Cartesian coordinates,
    Optics Express Vol. 26, Issue 15, pp. 18878-18896 (2018),
    <https://doi.org/10.1364/OE.26.018878>.
    """

    def __init__(self, X, scaling, symbolic=False):
        self.rc = {"classical": RCClassical, "normal": RCNormal}[scaling](symbolic)

        self.X = X
        self.L = 0
        self.last = [None, None]

    def __iter__(self):
        return self

    def __next__(self):
        if self.L == 0:
            out = numpy.array([0 * self.X[0] + self.rc.p0])
        else:
            alpha, beta, gamma = self.rc[self.L]

            shape = list(self.last[0].shape)
            shape[0] += 1
            out = numpy.zeros(shape, dtype=self.last[0].dtype)

            last_X = self.last[0] * self.X[0]
            last_Y = self.last[0] * self.X[1]

            out[:-1] += last_X + last_Y[::-1]
            out[1:] += last_X - last_Y[::-1]

            # It works without this seam correction, too.
            n = self.L + 1
            half = n // 2
            if n % 2 == 0:
                out[half - 1] -= last_X[half - 1]
                out[half] += last_Y[half - 1]
            else:
                out[half] += last_X[half]
                out[half] += last_Y[half - 1]

            if self.L > 1:
                out[1:-1] -= self.last[1]

        self.last[1] = self.last[0]
        self.last[0] = out
        self.L += 1
        return out


class RCClassical:
    def __init__(self, symbolic):
        self.p0 = 1

    def __getitem__(self, n):
        assert n > 0

        alpha = 1
        beta = 1
        gamma = None if n == 1 else 1

        return alpha, beta, gamma


class RCNormal:
    def __init__(self, symbolic):
        self.sqrt = sympy.sqrt if symbolic else math.sqrt
        pi = sympy.pi if symbolic else math.pi
        self.p0 = 1 / self.sqrt(pi)

    def __getitem__(self, n):
        assert n > 0

        alpha = 1
        beta = 1
        gamma = None if n == 1 else 1

        return alpha, beta, gamma
