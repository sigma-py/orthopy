import itertools
import math

import sympy

from ..tools import Iterator1D


def tree(n, *args, **kwargs):
    return list(itertools.islice(Iterator(*args, **kwargs), n + 1))


class Iterator(Iterator1D):
    def __init__(self, X, scaling, *args, **kwargs):
        cls = {"monic": RCMonic, "classical": RCClassical, "normal": RCNormal}[scaling]
        super().__init__(X, cls(*args, **kwargs))


class RCMonic:
    """Generate the recurrence coefficients a_k, b_k, c_k in

    P_{k+1}(x) = (a_k x - b_k)*P_{k}(x) - c_k P_{k-1}(x)

    for the Jacobi polynomials which are orthogonal on [-1, 1] with respect to the
    weight w(x)=[(1-x)^alpha]*[(1+x)^beta]; see
    <https://en.wikipedia.org/wiki/Jacobi_polynomials#Recurrence_relations>.
    """

    def __init__(self, alpha, beta, symbolic=False):
        self.alpha = alpha
        self.beta = beta
        self.gamma = sympy.gamma if symbolic else lambda x: math.gamma(float(x))

        self.frac = sympy.Rational if symbolic else lambda x, y: x / y
        self.nan = None if symbolic else math.nan

        self.p0 = 1

        # c[0] is not used in the actual recurrence, but is traditionally defined as the
        # integral of the weight function of the domain, i.e.,
        # ```
        # int_{-1}^{+1} (1-x)^a * (1+x)^b dx =
        #     2^(a+b+1) * Gamma(a+1) * Gamma(b+1) / Gamma(a+b+2).
        # ```
        # This is bad practice; the value could accidentally be used.

    def __getitem__(self, N):
        frac = self.frac
        alpha = self.alpha
        beta = self.beta

        a = 1

        if N == 0:
            b = frac(beta - alpha, alpha + beta + 2)
        else:
            b = frac(
                beta ** 2 - alpha ** 2,
                (2 * N + alpha + beta) * (2 * N + alpha + beta + 2),
            )

        # Note that we have the treat the case N==1 separately to avoid division by 0
        # for alpha=beta=-1/2.
        if N == 0:
            c = self.nan
        elif N == 1:
            c = frac(
                4 * (1 + alpha) * (1 + beta),
                (2 + alpha + beta) ** 2 * (3 + alpha + beta),
            )
        else:
            c = frac(
                4 * (N + alpha) * (N + beta) * N * (N + alpha + beta),
                (2 * N + alpha + beta) ** 2
                * (2 * N + alpha + beta + 1)
                * (2 * N + alpha + beta - 1),
            )
        return a, b, c


class RCClassical:
    def __init__(self, alpha, beta, symbolic=False):
        self.frac = sympy.Rational if symbolic else lambda x, y: x / y
        self.nan = None if symbolic else math.nan
        self.alpha = alpha
        self.beta = beta

        self.p0 = 1

        # gamma = sympy.gamma if symbolic else lambda x: math.gamma(float(x))
        # self.int_1 = (
        #     2 ** (alpha + beta + 1)
        #     * gamma(alpha + 1)
        #     * gamma(beta + 1)
        #     / gamma(alpha + beta + 2)
        # )

    def __getitem__(self, N):
        frac = self.frac
        alpha = self.alpha
        beta = self.beta

        # Treat N = 0 separately to avoid division by 0 for alpha = beta = -1/2.
        if N == 0:
            a = frac(alpha + beta + 2, 2)
            b = frac(beta - alpha, 2)
            c = self.nan
        else:
            a = frac(
                (2 * N + alpha + beta + 1) * (2 * N + alpha + beta + 2),
                2 * (N + 1) * (N + alpha + beta + 1),
            )
            b = frac(
                (beta ** 2 - alpha ** 2) * (2 * N + alpha + beta + 1),
                2 * (N + 1) * (N + alpha + beta + 1) * (2 * N + alpha + beta),
            )
            c = frac(
                (N + alpha) * (N + beta) * (2 * N + alpha + beta + 2),
                (N + 1) * (N + alpha + beta + 1) * (2 * N + alpha + beta),
            )

        return a, b, c


class RCNormal:
    def __init__(self, alpha, beta, symbolic=False):
        self.frac = sympy.Rational if symbolic else lambda x, y: x / y
        self.sqrt = sympy.sqrt if symbolic else math.sqrt
        self.nan = None if symbolic else math.nan
        self.alpha = alpha
        self.beta = beta

        gamma = sympy.gamma if symbolic else lambda x: math.gamma(float(x))
        self.int_1 = (
            2 ** (alpha + beta + 1)
            * gamma(alpha + 1)
            * gamma(beta + 1)
            / gamma(alpha + beta + 2)
        )
        self.p0 = self.sqrt(1 / self.int_1)

    def __getitem__(self, N):
        frac = self.frac
        sqrt = self.sqrt
        alpha = self.alpha
        beta = self.beta

        # Treat N==0 separately to avoid division by 0 for alpha=beta=-1/2
        # (Chebyshev 1).
        if N == 0:
            w = sqrt(frac(alpha + beta + 3, (alpha + 1) * (beta + 1)))
            a = frac(alpha + beta + 2, 2) * w
            b = frac(beta - alpha, 2) * w
        else:
            a = frac(2 * N + alpha + beta + 2, 2) * sqrt(
                frac(
                    (2 * N + alpha + beta + 1) * (2 * N + alpha + beta + 3),
                    (N + 1) * (N + alpha + 1) * (N + beta + 1) * (N + alpha + beta + 1),
                )
            )
            b = frac(beta ** 2 - alpha ** 2, 2 * (2 * N + alpha + beta)) * sqrt(
                frac(
                    (2 * N + alpha + beta + 3) * (2 * N + alpha + beta + 1),
                    (N + 1) * (N + alpha + 1) * (N + beta + 1) * (N + alpha + beta + 1),
                )
            )

        if N == 0:
            c = self.nan
        elif N == 1:
            c = frac(4 + alpha + beta, 2 + alpha + beta) * sqrt(
                frac(
                    (1 + alpha) * (1 + beta) * (5 + alpha + beta),
                    2 * (2 + alpha) * (2 + beta) * (2 + alpha + beta),
                )
            )
        else:
            c = frac(2 * N + alpha + beta + 2, 2 * N + alpha + beta) * sqrt(
                frac(
                    N
                    * (N + alpha)
                    * (N + beta)
                    * (N + alpha + beta)
                    * (2 * N + alpha + beta + 3),
                    (N + 1)
                    * (N + alpha + 1)
                    * (N + beta + 1)
                    * (N + alpha + beta + 1)
                    * (2 * N + alpha + beta - 1),
                )
            )

        return a, b, c
