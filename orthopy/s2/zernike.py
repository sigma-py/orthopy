import itertools

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
        self.rc = RCClassical(symbolic)

        self.X = X
        self.L = 0
        self.last = [None, None]

    def __iter__(self):
        return self

    def __next__(self):
        if self.L == 0:
            out = numpy.array([0 * self.X[0] + self.rc.p0])
        elif self.L == 1:
            # unfortunately, this needs to be treated as a separate case
            shape = list(self.last[0].shape)
            shape[0] += 1
            out = numpy.zeros(shape, dtype=self.last[0].dtype)

            last_X = self.last[0] * self.X[0]
            last_Y = self.last[0] * self.X[1]

            out[0] = last_Y[0]
            out[1] = last_X[0]
        else:
            alpha, beta, gamma, delta = self.rc[self.L]

            shape = list(self.last[0].shape)
            shape[0] += 1
            out = numpy.zeros(shape, dtype=self.last[0].dtype)

            last_X = self.last[0] * self.X[0]
            last_Y = self.last[0] * self.X[1]
            last_Y = last_Y[::-1]

            n = self.L + 1
            half = n // 2

            x_plus_y = last_X + last_Y
            x_minus_y = last_X - last_Y

            out[:-1] += x_plus_y
            out[1:] += x_minus_y

            # It works without this correction, too.
            if n % 2 == 0:
                out[half - 1] -= last_X[half - 1]
                out[half] += last_Y[half - 1]
            else:
                out[half] += x_plus_y[half]

            if self.L > 1:
                out[1:-1] -= self.last[1]

        self.last[1] = self.last[0]
        self.last[0] = out
        self.L += 1
        return out


class RCClassical:
    def __init__(self, symbolic, mu=1):
        self.S = sympy.S if symbolic else lambda x: x
        self.sqrt = numpy.vectorize(sympy.sqrt) if symbolic else numpy.sqrt
        self.mu = mu
        assert self.mu > -1

        self.p0 = 1

    def __getitem__(self, n):
        assert n > 0

        n = self.S(n)
        mu = self.mu

        alpha = numpy.full(n, (mu + 2 * n - 1) / (mu + n))
        # case distinction for mu==0
        if n == 1:
            beta = 1
        else:
            beta = (mu + 2 * n - 2) / (mu + n - 1)

        if n == 1:
            gamma = None
            delta = None
        else:  # n > 1
            k = numpy.arange(n - 1)
            gamma = (n - 1 - k) * (n + k + mu - 1) / ((mu + n) * (mu + n - 1))
            delta = (n - 1) / (mu + n - 1)

        return alpha, beta, gamma, delta
