import itertools
import math

import numpy
import sympy


def chebyshev1(n, standardization, symbolic=False):
    one_half = sympy.S(1) / 2 if symbolic else 0.5
    return jacobi(n, -one_half, -one_half, standardization, symbolic=symbolic)


def chebyshev2(n, standardization, symbolic=False):
    one_half = sympy.S(1) / 2 if symbolic else 0.5
    return jacobi(n, +one_half, +one_half, standardization, symbolic=symbolic)


def gegenbauer(n, lmbda, standardization, symbolic=False):
    return jacobi(n, lmbda, lmbda, standardization, symbolic=symbolic)


def legendre(n, standardization, symbolic=False):
    return jacobi(n, 0, 0, standardization, symbolic=symbolic)


def jacobi(n, alpha, beta, standardization, symbolic=False):
    """Generate the recurrence coefficients a_k, b_k, c_k in

    P_{k+1}(x) = (a_k x - b_k)*P_{k}(x) - c_k P_{k-1}(x)

    for the Jacobi polynomials which are orthogonal on [-1, 1] with respect to the
    weight w(x)=[(1-x)^alpha]*[(1+x)^beta]; see
    <https://en.wikipedia.org/wiki/Jacobi_polynomials#Recurrence_relations>.
    """
    if standardization == "monic":
        iterator = Monic(alpha, beta, symbolic)
        p0 = iterator.p0
    elif standardization == "p(1)=(n+alpha over n)" or (
        alpha == 0 and standardization == "p(1)=1"
    ):
        iterator = P1(alpha, beta, symbolic)
        p0 = iterator.p0
    else:
        assert (
            standardization == "normal"
        ), "Unknown standardization '{}'. (valid: {})".format(
            standardization, ", ".join(["monic", "p(1)=(n+alpha over n)", "normal"])
        )
        iterator = Normal(alpha, beta, symbolic)
        p0 = iterator.p0

    lst = list(itertools.islice(iterator, n))
    a = numpy.array([item[0] for item in lst])
    b = numpy.array([item[1] for item in lst])
    c = numpy.array([item[2] for item in lst])
    return p0, a, b, c


class Monic:
    def __init__(self, alpha, beta, symbolic):
        self.alpha = alpha
        self.beta = beta
        self.gamma = sympy.gamma if symbolic else lambda x: math.gamma(float(x))

        self.frac = sympy.Rational if symbolic else lambda x, y: x / y

        self.p0 = 1
        self.int_1 = (
            2 ** (alpha + beta + 1)
            * self.gamma(alpha + 1)
            * self.gamma(beta + 1)
            / self.gamma(alpha + beta + 2)
        )
        self.n = 0

    def __iter__(self):
        return self

    def __next__(self):
        frac = self.frac
        alpha = self.alpha
        beta = self.beta

        N = self.n

        a = 1

        if N == 0:
            b = frac(beta - alpha, alpha + beta + 2)
        else:
            b = frac(
                beta ** 2 - alpha ** 2,
                (2 * N + alpha + beta) * (2 * N + alpha + beta + 2),
            )

        # c[0] is not used in the actual recurrence, but is often defined as the
        # integral of the weight function of the domain, i.e.,
        # ```
        # int_{-1}^{+1} (1-x)^a * (1+x)^b dx =
        #     2^(a+b+1) * Gamma(a+1) * Gamma(b+1) / Gamma(a+b+2).
        # ```
        # Note also that we have the treat the case N==1 separately to avoid division by
        # 0 for alpha=beta=-1/2.
        if N == 0:
            c = self.int_1
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
        self.n += 1
        return a, b, c


class P1:
    def __init__(self, alpha, beta, symbolic):
        self.alpha = alpha
        self.beta = beta
        self.gamma = sympy.gamma if symbolic else lambda x: math.gamma(float(x))

        self.frac = sympy.Rational if symbolic else lambda x, y: x / y

        self.p0 = 1
        self.int_1 = (
            2 ** (alpha + beta + 1)
            * self.gamma(alpha + 1)
            * self.gamma(beta + 1)
            / self.gamma(alpha + beta + 2)
        )

        self.n = 0

    def __iter__(self):
        return self

    def __next__(self):
        frac = self.frac
        alpha = self.alpha
        beta = self.beta

        N = self.n

        # Treat N = 0 separately to avoid division by 0 for alpha = beta = -1/2.
        if N == 0:
            a = frac(alpha + beta + 2, 2)
        else:
            a = frac(
                (2 * N + alpha + beta + 1) * (2 * N + alpha + beta + 2),
                2 * (N + 1) * (N + alpha + beta + 1),
            )

        if N == 0:
            b = frac(beta - alpha, 2)
        else:
            b = frac(
                (beta ** 2 - alpha ** 2) * (2 * N + alpha + beta + 1),
                2 * (N + 1) * (N + alpha + beta + 1) * (2 * N + alpha + beta),
            )

        if N == 0:
            c = self.int_1
        else:
            c = frac(
                (N + alpha) * (N + beta) * (2 * N + alpha + beta + 2),
                (N + 1) * (N + alpha + beta + 1) * (2 * N + alpha + beta),
            )

        self.n += 1
        return a, b, c


class Normal:
    def __init__(self, alpha, beta, symbolic):
        self.alpha = alpha
        self.beta = beta
        self.gamma = sympy.gamma if symbolic else lambda x: math.gamma(float(x))

        self.frac = sympy.Rational if symbolic else lambda x, y: x / y
        self.sqrt = sympy.sqrt if symbolic else numpy.sqrt
        self.int_1 = (
            2 ** (alpha + beta + 1)
            * self.gamma(alpha + 1)
            * self.gamma(beta + 1)
            / self.gamma(alpha + beta + 2)
        )
        self.p0 = self.sqrt(1 / self.int_1)

        self.n = 0

    def __iter__(self):
        return self

    def __next__(self):
        frac = self.frac
        alpha = self.alpha
        beta = self.beta
        sqrt = self.sqrt

        N = self.n
        int_1 = self.int_1

        # Treat N==0 separately to avoid division by 0 for alpha=beta=-1/2
        # (Chebyshev 1).
        if N == 0:
            a = frac(alpha + beta + 2, 2) * sqrt(
                frac(alpha + beta + 3, (alpha + 1) * (beta + 1))
            )
        else:
            a = frac(2 * N + alpha + beta + 2, 2) * sqrt(
                frac(
                    (2 * N + alpha + beta + 1) * (2 * N + alpha + beta + 3),
                    (N + 1) * (N + alpha + 1) * (N + beta + 1) * (N + alpha + beta + 1),
                )
            )

        # Treat N==0 separately to avoid division by 0 for alpha=beta=-1/2
        # (Chebyshev 1).
        if N == 0:
            b = frac(beta - alpha, 2) * sqrt(
                frac(alpha + beta + 3, (alpha + 1) * (beta + 1))
            )
        else:
            b = frac(beta ** 2 - alpha ** 2, 2 * (2 * N + alpha + beta)) * sqrt(
                frac(
                    (2 * N + alpha + beta + 3) * (2 * N + alpha + beta + 1),
                    (N + 1) * (N + alpha + 1) * (N + beta + 1) * (N + alpha + beta + 1),
                )
            )

        if N == 0:
            c = int_1
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

        self.n += 1
        return a, b, c
