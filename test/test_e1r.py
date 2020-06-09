import itertools

import pytest
import sympy
from sympy import gamma

import orthopy

# def _integrate(f, alpha, x):
#     return sympy.integrate(f * x ** alpha * sympy.exp(-x), (x, 0, +oo))


# This function returns the integral of all monomials up to a given degree.
def _integrate_all_monomials(max_k, alpha):
    out = []
    for k in range(max_k + 1):
        if k == 0:
            out.append(gamma(alpha + 1))
        else:
            out.append(out[k - 1] * (alpha + k))
    return out


# Integrating polynomials is easily done by integrating the individual monomials and
# summing.
def _integrate_poly(p, alpha, x):
    coeffs = p.all_coeffs()[::-1]
    int_all_monomials = _integrate_all_monomials(p.degree(), alpha)
    return sum(coeff * mono_int for coeff, mono_int in zip(coeffs, int_all_monomials))


@pytest.mark.parametrize("alpha", [0, 1])
def test_integral0(alpha, n=4):
    x = sympy.Symbol("x")
    p = sympy.poly(x)
    vals = orthopy.e1r.tree(n, p, alpha=alpha, scaling="normal", symbolic=True)

    assert _integrate_poly(vals[0], alpha, x) == 1
    for val in vals[1:]:
        assert _integrate_poly(val, alpha, x) == 0


@pytest.mark.parametrize("alpha", [0, 1])
@pytest.mark.parametrize("scaling", ["monic", "classical", "normal"])
def test_orthogonality(alpha, scaling, n=4):
    x = sympy.Symbol("x")
    p = sympy.poly(x)
    tree = orthopy.e1r.tree(n, p, scaling, alpha=alpha, symbolic=True)
    for f0, f1 in itertools.combinations(tree, 2):
        assert _integrate_poly(f0 * f1, alpha, x) == 0


@pytest.mark.parametrize("alpha", [0, 1])
def test_normality(alpha, n=4):
    x = sympy.Symbol("x")
    p = sympy.poly(x)
    tree = orthopy.e1r.tree(n, p, "normal", alpha=alpha, symbolic=True)
    tree[0] = sympy.poly(tree[0], x)
    for val in tree:
        assert _integrate_poly(val ** 2, alpha, x) == 1


def test_show():
    orthopy.e1r.show(L=4)


if __name__ == "__main__":
    test_show()
    # import matplotlib.pyplot as plt
    # plt.show()
    # plt.savefig("e1r.png", transparent=True)
