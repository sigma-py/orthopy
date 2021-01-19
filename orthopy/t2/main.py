import numpy as np
import sympy


class Eval:
    """Evaluates orthogonal polynomials on the triangle.

    The return value is a list of arrays, where `out[k]` hosts the `2*k+1` values of the
    `k`th level of the tree

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

    def __init__(self, bary, scaling, symbolic="auto"):
        if symbolic == "auto":
            symbolic = np.asarray(bary).dtype == sympy.Basic

        self.bary = bary

        self.k = 0
        self.last = [None, None]

        self.rc = {"classical": RCClassical, "monic": RCMonic, "normal": RCNormal}[
            scaling
        ](symbolic)

        self.int_p0 = self.rc.p0

    def __iter__(self):
        return self

    def __next__(self):
        u, v, w = self.bary

        if self.k == 0:
            out = np.array([u * 0 + self.rc.p0])
        else:
            alpha, beta, gamma, delta, epsilon = self.rc[self.k]
            out = np.concatenate(
                [
                    self.last[0] * (np.multiply.outer(alpha, 1 - 2 * w).T - beta).T,
                    [delta * self.last[0][-1] * (u - v)],
                ]
            )

            if self.k > 1:
                out[:-2] -= (self.last[1].T * gamma).T
                out[-1] -= epsilon * self.last[1][-1] * (u + v) ** 2

        self.last[1] = self.last[0]
        self.last[0] = out
        self.k += 1
        return out


class RCClassical:
    def __init__(self, symbolic):
        self.S = sympy.S if symbolic else lambda x: x
        self.p0 = 1

    def __getitem__(self, n):
        n = self.S(n)

        r = np.arange(n)
        alpha = n * (2 * n + 1) / ((n - r) * (n + r + 1))
        beta = n * (2 * r + 1) ** 2 / ((n - r) * (n + r + 1) * (2 * n - 1))
        delta = (2 * n - 1) / n

        if n in [0, 1]:
            gamma = None
            epsilon = None
        else:
            r = np.arange(n - 1)
            gamma = (
                (n - r - 1)
                * (n + r)
                * (2 * n + 1)
                / ((n - r) * (n + r + 1) * (2 * n - 1))
            )
            epsilon = (n - 1) / n
        return alpha, beta, gamma, delta, epsilon


class RCMonic:
    def __init__(self, symbolic):
        self.S = sympy.S if symbolic else lambda x: x

        self.p0 = 1

    def __getitem__(self, n):
        assert n > 0

        r = np.arange(n)
        alpha = np.ones(n, dtype=int)
        delta = 1
        n = self.S(n)

        beta = (2 * r + 1) ** 2 / ((2 * n - 1) * (2 * n + 1))

        if n == 1:
            gamma = None
            epsilon = None
        else:
            r = np.arange(n - 1)
            gamma = (n - r - 1) ** 2 * (n + r) ** 2 / ((2 * n - 1) ** 2 * (n - 1) * n)
            epsilon = (n - 1) ** 2 / ((2 * n - 1) * (2 * n - 3))

        return alpha, beta, gamma, delta, epsilon


class RCNormal:
    """
    The coefficients here are based on the insight that

        int_T P_{n, r}^2 =
            int_0^1 L_r^2(t) dt * int_0^1 q_{n,r}(w)^2 (1-w)^(r+s+1) dw.

    For reference, see page 219 in

        Farouki, Goodman, Sauer,
        Construction of orthogonal bases for polynomials in Bernstein form on triangular
        and simplex domains,
        Computer Aided Geometric Design 20 (2003) 209–230,

    and the reference to Gould, 1972, there.

    The Legendre integral is 1/(2*r+1), and one gets

      int_T P_{n, r}^2 = 1 / (2*r+1) / (2*n+2)
                         * sum_{i=0}^{n-r} sum_{j=0}^{n-r}
                              (-1)**(i+j) * binom(n+r+1, i) * binom(n-r, i)
                                          * binom(n+r+1, j) * binom(n-r, j)
                                          / binom(2*n+1, i+j)

    Astonishingly, the double sum is always 1, hence

      int_T P_{n, r}^2 = 1 / (2*r+1) / (2*n+2).
    """

    def __init__(self, symbolic):
        self.S = sympy.S if symbolic else lambda x: x
        self.sqrt = sympy.sqrt if symbolic else np.sqrt

        self.p0 = self.sqrt(2)

    def __getitem__(self, n):
        n = self.S(n)
        sqrt = self.sqrt

        r = np.arange(n)
        alpha = sqrt((n + 1) * n) * ((2 * n + 1) / ((n - r) * (n + r + 1)))
        beta = (
            sqrt((n + 1) * n) * (2 * r + 1) ** 2 / ((n - r) * (n + r + 1) * (2 * n - 1))
        )
        delta = sqrt((2 * n + 1) * (n + 1) * (2 * n - 1) / n ** 3)

        if n in [0, 1]:
            gamma = None
            epsilon = None
        else:
            r = np.arange(n - 1)
            gamma = sqrt((n + 1) / (n - 1)) * (
                (n - r - 1)
                * (n + r)
                * (2 * n + 1)
                / ((n - r) * (n + r + 1) * (2 * n - 1))
            )
            epsilon = sqrt((2 * n + 1) * (n + 1) * (n - 1) / ((2 * n - 3) * n ** 2))

        return alpha, beta, gamma, delta, epsilon
