import ndim


def _integrate_poly(p):
    return sum(
        c * ndim.nball.integrate_monomial(k, symbolic=True)
        for c, k in zip(p.coeffs(), p.monoms())
    )
