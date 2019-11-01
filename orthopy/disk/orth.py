import itertools
import numpy
import sympy


def tree(X, n, symbolic=False):
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
    return list(itertools.islice(Orth(X, symbolic), n + 1))


class Orth:
    def __init__(self, X, symbolic=False):
        self.frac = sympy.Rational if symbolic else lambda x, y: x / y
        self.sqrt = sympy.sqrt if symbolic else numpy.sqrt
        self.pi = sympy.pi if symbolic else numpy.pi

        self.mu = self.frac(1, 2)

        self.p0 = 1 / self.sqrt(self.pi)
        self.X = X
        self.L = 0
        self.last = [None, None]

    def alpha(self, n):
        return numpy.array(
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

    def beta(self, n):
        return 2 * self.sqrt(
            self.frac(
                (n + self.mu - 1) * (n + self.mu + self.frac(1, 2)),
                (n + 2 * self.mu - 1) * n,
            )
        )

    def gamma(self, n):
        return numpy.array(
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

    def delta(self, n):
        return self.sqrt(
            self.frac(
                (n - 1)
                * (n + 2 * self.mu - 2)
                * (n + self.mu - self.frac(1, 2))
                * (n + self.mu + self.frac(1, 2)),
                n * (n + 2 * self.mu - 1) * (n + self.mu - 1) * (n + self.mu - 2),
            )
        )

    def __iter__(self):
        return self

    def __next__(self):
        one_min_x2 = 1 - self.X[0] ** 2

        if self.L == 0:
            out = numpy.array([0 * self.X[0] + self.p0])
        else:
            out = numpy.concatenate(
                [
                    self.last[0] * numpy.multiply.outer(self.alpha(self.L), self.X[0]),
                    [self.last[0][-1] * self.beta(self.L) * self.X[1]],
                ]
            )

            if self.L > 1:
                out[:-2] -= (self.last[1].T * self.gamma(self.L)).T
                out[-1] -= self.last[1][-1] * self.delta(self.L) * one_min_x2

        self.last[1] = self.last[0]
        self.last[0] = out
        self.L += 1
        return out
