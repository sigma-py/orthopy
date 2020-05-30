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

    def __init__(self, X, symbolic=False):
        self.rc = RCNormal(symbolic)

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


class RCNormal:
    """Recurrence coefficients for the disk with the Gegenbauer-style weight function
    $$
    \\frac{\\mu + 1/2}{\\pi} (1 - x^2 - y^2)^{\\mu - 1/2}
    $$
    scaled for normality.
    """

    def __init__(self, symbolic, mu=None):
        self.frac = sympy.Rational if symbolic else lambda x, y: x / y
        self.sqrt = sympy.sqrt if symbolic else numpy.sqrt
        self.mu = self.frac(1, 2)
        # default: weight function 1
        self.mu = self.frac(1, 2) if mu is None else mu
        assert self.mu > -self.frac(1, 2)

        pi = sympy.pi if symbolic else numpy.pi
        self.p0 = 1 / self.sqrt(pi)

    def __getitem__(self, n):
        alpha = numpy.array(
            [
                2
                * self.sqrt(
                    self.frac(
                        (n + self.mu + self.frac(1, 2))
                        * (n + self.mu - self.frac(1, 2)),
                        (n - k) * (n + k + 2 * self.mu),
                    )
                )
                for k in range(n)
            ]
        )
        beta = 2 * self.sqrt(
            self.frac(
                (n + self.mu - 1) * (n + self.mu + self.frac(1, 2)),
                (n + 2 * self.mu - 1) * n,
            )
        )
        if n in [0, 1]:
            gamma = None
            delta = None
        else:  # n > 1
            gamma = numpy.array(
                [
                    self.sqrt(
                        self.frac(
                            (n - 1 - k)
                            * (n + self.mu + self.frac(1, 2))
                            * (n + k + 2 * self.mu - 1),
                            (n - k)
                            * (n + self.mu - self.frac(3, 2))
                            * (n + k + 2 * self.mu),
                        )
                    )
                    for k in range(n - 1)
                ]
            )
            # case distinction to avoid undefined expressions for mu = 0.
            if n == 2:
                delta = self.sqrt(
                    self.frac(
                        (self.frac(3, 2) + self.mu) * (self.frac(5, 2) + self.mu),
                        (1 + 2 * self.mu) * (1 + self.mu),
                    )
                )
            else:
                delta = self.sqrt(
                    self.frac(
                        (n - 1)
                        * (n + 2 * self.mu - 2)
                        * (n + self.mu - self.frac(1, 2))
                        * (n + self.mu + self.frac(1, 2)),
                        n
                        * (n + 2 * self.mu - 1)
                        * (n + self.mu - 1)
                        * (n + self.mu - 2),
                    )
                )
        return alpha, beta, gamma, delta
