import itertools

import numpy
import sympy


def tree(bary, n, scaling, symbolic=False):
    """Evaluates the entire tree of orthogonal triangle polynomials.

    The return value is a list of arrays, where `out[k]` hosts the `2*k+1`
    values of the `k`th level of the tree

        (0, 0)
        (0, 1)   (1, 1)
        (0, 2)   (1, 2)   (2, 2)
          ...      ...      ...

    For reference, see

    Abedallah Rababah,
    Recurrence Relations for Orthogonal Polynomials on Triangular Domains,
    Mathematics 2016, 4(2), 25,
    <https://doi.org/10.3390/math4020025>.

    (The formulation there is more complicated than necessary, however, and doesn't
    include the normalization.)
    """
    return list(itertools.islice(Iterator(bary, scaling, symbolic), n + 1))


class Iterator:
    def __init__(self, bary, scaling, symbolic=False):
        S = numpy.vectorize(sympy.S) if symbolic else lambda x: x
        sqrt = numpy.vectorize(sympy.sqrt) if symbolic else numpy.sqrt

        self.bary = bary

        self.k = 0
        self.last = [None, None]

        if scaling == "1":
            self.p0 = 1

            def alpha(n):
                r = numpy.arange(n)
                return S(n * (2 * n + 1)) / ((n - r) * (n + r + 1))

            def beta(n):
                r = numpy.arange(n)
                return S(n * (2 * r + 1) ** 2) / ((n - r) * (n + r + 1) * (2 * n - 1))

            def gamma(n):
                r = numpy.arange(n - 1)
                return S((n - r - 1) * (n + r) * (2 * n + 1)) / (
                    (n - r) * (n + r + 1) * (2 * n - 1)
                )

            def delta(n):
                return S(2 * n - 1) / n

            def epsilon(n):
                return S(n - 1) / n

        else:
            # The coefficients here are based on the insight that
            #
            #   int_T P_{n, r}^2 =
            #       int_0^1 L_r^2(t) dt * int_0^1 q_{n,r}(w)^2 (1-w)^(r+s+1) dw.
            #
            # For reference, see page 219 in
            #
            #  Farouki, Goodman, Sauer,
            #  Construction of orthogonal bases for polynomials in Bernstein form on
            #  triangular and simplex domains,
            #  Computer Aided Geometric Design 20 (2003) 209–230,
            #
            # and the reference to Gould, 1972, there.
            #
            # The Legendre integral is 1/(2*r+1), and one gets
            #
            #   int_T P_{n, r}^2 = 1 / (2*r+1) / (2*n+2)
            #       sum_{i=0}^{n-r} sum_{j=0}^{n-r}
            #           (-1)**(i+j) * binom(n+r+1, i) * binom(n-r, i)
            #                       * binom(n+r+1, j) * binom(n-r, j)
            #                       / binom(2*n+1, i+j)
            #
            # Astonishingly, the double sum is always 1, hence
            #
            #   int_T P_{n, r}^2 = 1 / (2*r+1) / (2*n+2).
            #
            assert scaling == "normal"
            self.p0 = sqrt(2)

            def alpha(n):
                r = numpy.arange(n)
                return sqrt((n + 1) * n) * (S(2 * n + 1) / ((n - r) * (n + r + 1)))

            def beta(n):
                r = numpy.arange(n)
                return (
                    sqrt((n + 1) * n)
                    * S((2 * r + 1) ** 2)
                    / ((n - r) * (n + r + 1) * (2 * n - 1))
                )

            def gamma(n):
                r = numpy.arange(n - 1)
                return sqrt(S(n + 1) / (n - 1)) * (
                    S((n - r - 1) * (n + r) * (2 * n + 1))
                    / ((n - r) * (n + r + 1) * (2 * n - 1))
                )

            def delta(n):
                return sqrt(S((2 * n + 1) * (n + 1) * (2 * n - 1)) / n ** 3)

            def epsilon(n):
                return sqrt(S((2 * n + 1) * (n + 1) * (n - 1)) / ((2 * n - 3) * n ** 2))

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.epsilon = epsilon

    def __iter__(self):
        return self

    def __next__(self):
        u, v, w = self.bary
        L = self.k

        if self.k == 0:
            out = numpy.array([numpy.zeros_like(u) + self.p0])
        else:
            out = numpy.concatenate(
                [
                    self.last[0]
                    * (
                        numpy.multiply.outer(self.alpha(L), 1 - 2 * w).T - self.beta(L)
                    ).T,
                    [self.delta(L) * self.last[0][L - 1] * (u - v)],
                ]
            )

            if L > 1:
                out[:-2] -= (self.last[1].T * self.gamma(L)).T
                out[-1] -= self.epsilon(L) * self.last[1][-1] * (u + v) ** 2

        self.last[1] = self.last[0]
        self.last[0] = out
        self.k += 1
        return out
