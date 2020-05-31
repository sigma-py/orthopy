import itertools

import numpy
import sympy


def tree(n, *args, **kwargs):
    return list(itertools.islice(Iterator(*args, **kwargs), n + 1))


class Iterator:
    """Evaluates the entire tree of orthogonal polynomials on the unit disk.

    The return value is a list of arrays, where `out[k]` hosts the `2*k+1`
    values of the `k`th level of the tree

        (0, 0)
        (0, 1)   (1, 1)
        (0, 2)   (1, 2)   (2, 2)
          ...      ...      ...

    See

    Yuan Xu
    Orthogonal polynomials of several variables,
    Jan. 2017,
    <https://arxiv.org/abs/1701.02709>

    equation (3.4) for a formulation in terms of Gegenbauer polynomials C. The
    recurrence relation can be worked out from there.
    """

    def __init__(self, X, scaling, symbolic=False):
        self.rc = {"classical": RCClassical, "monic": RCMonic, "normal": RCNormal}[
            scaling
        ](symbolic)

        self.X = X
        self.one_min_x2 = 1 - self.X[0] ** 2
        self.L = 0
        self.last = [None, None]

    def __iter__(self):
        return self

    def __next__(self):
        if self.L == 0:
            out = numpy.array([0 * self.X[0] + self.rc.p0])
        else:
            alpha, beta, gamma, delta = self.rc[self.L]
            out = numpy.concatenate(
                [
                    self.last[0] * numpy.multiply.outer(alpha, self.X[0]),
                    [self.last[0][-1] * beta * self.X[1]],
                ]
            )

            if self.L > 1:
                out[:-2] -= (self.last[1].T * gamma).T
                out[-1] -= self.last[1][-1] * delta * self.one_min_x2

        self.last[1] = self.last[0]
        self.last[0] = out
        self.L += 1
        return out


class RCClassical:
    """The maximum values (which are attained at (1, 0) for the first and (0, 1) for the
    last polynomial in each level) is 1.
    """

    def __init__(self, symbolic, mu=1):
        self.S = sympy.S if symbolic else lambda x: x
        self.sqrt = numpy.vectorize(sympy.sqrt) if symbolic else numpy.sqrt
        self.mu = mu
        assert self.mu > -1

        self.p0 = 1

    # How to get from the monic representation to this one:
    # The value at (0, 1) needs to be 1, so
    #
    # self.last[0][-1] * beta * self.X[1] - self.last[1][-1] * delta * self.one_min_x2
    # = beta - delta
    #
    # (from the recurrence) needs to be 1. Hence
    #
    #  beta_new = beta / (beta - delta)
    #  delta_new = delta / (beta - delta) / beta_min_delta_from_previous_step
    #
    # Calling z = beta - delta, this establishes the formula
    #
    #     z(n) = beta(n) - delta(n) / z(n-1)
    #
    # which is a continued fraction. Using sympy, it can be shown that
    #
    #   z(n) = (mu + n - 1) / (mu + 2 * (n - 1))
    #
    # so
    #
    #   beta(n) = (mu + 2 * (n - 1)) / (mu + n - 1)
    #
    # and delta accordingly.
    #
    def __getitem__(self, n):
        n = self.S(n)
        mu = self.mu

        alpha = numpy.ones(n, dtype=int)
        # case distinction for mu==0
        if n == 1:
            beta = 1
        else:
            beta = (mu + 2 * (n - 1)) / (mu + n - 1)

        if n == 1:
            gamma = None
            delta = None
        else:  # n > 1
            k = numpy.arange(n - 1)
            gamma = (
                (n - 1 - k) * (n + k + mu - 1) / ((2 * n + mu - 3) * (2 * n + mu - 1))
            )
            delta = (n - 1) / (mu + n - 1)

        return alpha, beta, gamma, delta


class RCMonic:
    """alpha and beta both equal 1.
    """

    def __init__(self, symbolic, mu=1):
        self.S = sympy.S if symbolic else lambda x: x
        self.sqrt = numpy.vectorize(sympy.sqrt) if symbolic else numpy.sqrt
        self.mu = mu
        assert self.mu > -1

        self.p0 = 1

    def __getitem__(self, n):
        n = self.S(n)
        mu = self.mu

        alpha = numpy.ones(n, dtype=int)
        beta = 1

        if n == 1:
            gamma = None
            delta = None
        else:  # n > 1
            k = numpy.arange(n - 1)
            gamma = (
                (n - 1 - k) * (n + k + mu - 1) / ((2 * n + mu - 3) * (2 * n + mu - 1))
            )
            # case distinction to avoid undefined expression for mu = 0.
            if n == 2:
                delta = self.S(1) / (mu + 2)
            else:
                delta = (n - 1) * (n + mu - 2) / ((2 * n + mu - 2) * (2 * n + mu - 4))

        return alpha, beta, gamma, delta


class RCNormal:
    """Recurrence coefficients for the disk with the Gegenbauer-style weight function
    $$
    \\frac{\\mu + 1}{2\\pi} (1 - x^2 - y^2)^{(\\mu - 1)/2}
    $$
    scaled for normality.

    In some other tests, one finds a rescaled mu.
    $$
    \\frac{\\mu + 1/2}{\\pi} (1 - x^2 - y^2)^{\\mu - 1/2}
    $$
    """

    # default: weight function 1
    def __init__(self, symbolic, mu=1):
        self.S = sympy.S if symbolic else lambda x: x
        self.sqrt = numpy.vectorize(sympy.sqrt) if symbolic else numpy.sqrt
        self.mu = mu
        assert self.mu > -1

        pi = sympy.pi if symbolic else numpy.pi
        self.p0 = 1 / self.sqrt(pi)

    def __getitem__(self, n):
        n = self.S(n)
        mu = self.mu

        k = numpy.arange(n)
        alpha = self.sqrt(
            (2 * n + mu + 1) * (2 * n + mu - 1) / ((n - k) * (n + k + mu))
        )

        beta = self.sqrt((2 * n + mu - 2) * (2 * n + mu + 1) / (n * (n + mu - 1)))

        if n == 1:
            gamma = None
            delta = None
        else:  # n > 1
            k = numpy.arange(n - 1)
            gamma = self.sqrt(
                (n - 1 - k)
                * (2 * n + mu + 1)
                * (n + k + mu - 1)
                / ((n - k) * (2 * n + mu - 3) * (n + k + mu))
            )
            # case distinction to avoid undefined expression for mu = 0.
            if n == 2:
                two = self.S(2)
                delta = self.sqrt((3 + mu) * (5 + mu) / (two * (1 + mu) * (2 + mu)))
            else:
                delta = self.sqrt(
                    (n - 1)
                    * (n + mu - 2)
                    * (2 * n + mu - 1)
                    * (2 * n + mu + 1)
                    / (n * (n + mu - 1) * (2 * n + mu - 2) * (2 * n + mu - 4))
                )
        return alpha, beta, gamma, delta
