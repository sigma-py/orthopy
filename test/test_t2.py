import itertools

import numpy
import pytest
import scipy.special
import sympy
from sympy import Rational, S

import ndim
import orthopy
from helpers import get_nth

b0 = sympy.Symbol("b0")
b1 = sympy.Symbol("b1")


# def _integrate(f):
#     return sympy.integrate(f, (b0, 0, 1 - b1), (b1, 0, 1))


def _integrate_poly(p):
    return sum(
        c * ndim.nsimplex.integrate_monomial(k, symbolic=True)
        for c, k in zip(p.coeffs(), p.monoms())
    )


def op(i, j, x, y):
    # scaling = "monic"
    scaling = "classical"

    iterator = orthopy.c1.jacobi.Eval((x - y) / (x + y), 0, 0, scaling)
    val1 = get_nth(iterator, i)

    # val1 = numpy.polyval(scipy.special.jacobi(i, 0, 0), (x - y) / (x + y))

    # treat x==0, y==0 separately
    if isinstance(val1, numpy.ndarray):
        idx = numpy.where(numpy.logical_and(x == 0, y == 0))[0]
        val1[idx] = numpy.polyval(scipy.special.jacobi(i, 0, 0), 0.0)
    else:
        if numpy.isnan(val1):
            val1 = numpy.polyval(scipy.special.jacobi(i, 0, 0), 0.0)

    iterator = orthopy.c1.jacobi.Eval(1 - 2 * (x + y), 2 * i + 1, 0, scaling)
    val2 = get_nth(iterator, j)
    # val2 = numpy.polyval(scipy.special.jacobi(j, 2*i+1, 0), 1-2*(x+y))

    flt = numpy.vectorize(float)
    return flt(
        numpy.sqrt(2 * i + 1)
        * val1
        * (x + y) ** i
        * numpy.sqrt(2 * j + 2 * i + 2)
        * val2
    )


def eval_orthpolys4(bary):
    """Evaluate all orthogonal polynomials at x.
    See, e.g.,

    S.-A. Papanicolopulos,
    New fully symmetric and rotationally symmetric cubature rules on the triangle using
    minimal orthonormal bases,
    <https://arxiv.org/pdf/1411.5631.pdf>.
    """
    x, y = bary[0], bary[1]

    def f(i, j):
        return op(i, j, x, y)

    return [
        [+f(0, 0)],
        [-f(0, 1), +f(1, 0)],
        [+f(0, 2), -f(1, 1), +f(2, 0)],
        [-f(0, 3), +f(1, 2), -f(2, 1), +f(3, 0)],
        [+f(0, 4), -f(1, 3), +f(2, 2), -f(3, 1), +f(4, 0)],
    ]


@pytest.mark.parametrize("x", [numpy.array([0.24, 0.65]), numpy.random.rand(2, 5)])
def test_t2_orth(x, tol=1.0e-12):
    L = 4
    exacts = eval_orthpolys4(x)

    bary = numpy.array([x[0], x[1], 1 - x[0] - x[1]])
    vals = orthopy.t2.tree(L, bary, "normal")

    for val, ex in zip(vals, exacts):
        for v, e in zip(val, ex):
            assert numpy.all(numpy.abs(v - e) < tol * numpy.abs(e))


def test_t2_orth_exact():
    x = numpy.array([S(1) / 3, S(1) / 7])

    L = 2
    exacts = [
        [sympy.sqrt(2)],
        [-S(8) / 7, 8 * sympy.sqrt(3) / 21],
        [
            -197 * sympy.sqrt(6) / 441,
            -136 * sympy.sqrt(2) / 147,
            -26 * sympy.sqrt(30) / 441,
        ],
    ]

    bary = numpy.array([x[0], x[1], 1 - x[0] - x[1]])
    vals = orthopy.t2.tree(L, bary, "normal", symbolic=True)

    for val, ex in zip(vals, exacts):
        for v, e in zip(val, ex):
            assert v == e


def test_t2_orth_classical_exact():
    x = numpy.array([[S(1) / 5, S(2) / 5, S(3) / 5], [S(1) / 7, S(2) / 7, S(3) / 7]])

    L = 2
    exacts = [
        [[1, 1, 1]],
        [[-S(34) / 35, S(2) / 35, S(38) / 35], [S(2) / 35, S(4) / 35, S(6) / 35]],
    ]

    bary = numpy.array([x[0], x[1], 1 - x[0] - x[1]])
    vals = orthopy.t2.tree(L, bary, "classical", symbolic=True)

    for val, ex in zip(vals, exacts):
        for v, e in zip(val, ex):
            assert numpy.all(v == e)


@pytest.mark.parametrize(
    "scaling,int0",
    [
        ("classical", Rational(1, 2)),
        ("monic", Rational(1, 2)),
        ("normal", sympy.sqrt(2) / 2),
    ],
)
def test_integral0(scaling, int0, n=4):
    p = [sympy.poly(x, [b0, b1]) for x in [b0, b1, 1 - b0 - b1]]
    # b = [b0, b1, 1 - b0 - b1]

    iterator = orthopy.t2.Eval(p, scaling, symbolic=True)
    for k, vals in enumerate(itertools.islice(iterator, n)):
        if k == 0:
            assert _integrate_poly(vals[0]) == int0
        else:
            for val in vals:
                assert _integrate_poly(val) == 0


def test_normality(n=4):
    p = [sympy.poly(x, [b0, b1]) for x in [b0, b1, 1 - b0 - b1]]
    # b = [b0, b1, 1 - b0 - b1]
    iterator = orthopy.t2.Eval(p, "normal", symbolic=True)
    for vals in itertools.islice(iterator, n):
        for val in vals:
            assert _integrate_poly(val ** 2) == 1


@pytest.mark.parametrize("scaling", ["classical", "monic", "normal"])
def test_orthogonality(scaling, n=3):
    p = [sympy.poly(x, [b0, b1]) for x in [b0, b1, 1 - b0 - b1]]
    # b = [b0, b1, 1 - b0 - b1]
    tree = numpy.concatenate(orthopy.t2.tree(n, p, scaling, symbolic=True))
    for f0, f1 in itertools.combinations(tree, 2):
        assert _integrate_poly(f0 * f1) == 0


def test_show_single(degrees=(1, 1)):
    orthopy.t2.show_single(degrees, colorbar=False)
    orthopy.t2.savefig_single("triangle.png", degrees, colorbar=False)


if __name__ == "__main__":
    # x_ = numpy.array([0.24, 0.65])
    # # x_ = numpy.random.rand(3, 2)
    # test_t2_orth(x=x_)
    test_show_single((2, 1))
