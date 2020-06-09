import numpy
import sympy


def volume_nball(n, symbolic, r=1):
    pi = sympy.pi if symbolic else numpy.pi

    if n == 0:
        return 1
    elif n == 1:
        return 2 * r
    return volume_nball(n - 2, symbolic, r=r) * 2 * pi / n * r ** 2


def _integrate_monomial(k, symbolic=True, r=1):
    frac = sympy.Rational if symbolic else lambda a, b: a / b
    if any(a % 2 == 1 for a in k):
        return 0

    n = len(k)
    if all(a == 0 for a in k):
        return volume_nball(n, symbolic, r=r)

    # find first nonzero
    idx = next(i for i, j in enumerate(k) if j > 0)
    alpha = frac((k[idx] - 1) * r ** 2, sum(k) + n)
    k2 = k.copy()
    k2[idx] -= 2
    return _integrate_monomial(k2, symbolic, r=r) * alpha


def _integrate_poly(p):
    return sum(c * _integrate_monomial(list(k)) for c, k in zip(p.coeffs(), p.monoms()))
