import math

import numpy as np
from scipy.special import legendre

import orthopy


def test_clenshaw(tol=1.0e-14):
    n = 5
    rc = orthopy.c1.jacobi.RecurrenceCoefficients("monic", 0, 0, symbolic=False)
    _, alpha, beta = np.array([rc[k] for k in range(n)]).T

    t = 1.0

    a = np.ones(n + 1)
    value = orthopy.c1.clenshaw(a, alpha, beta, t)

    ref = math.fsum([np.polyval(legendre(i, monic=True), t) for i in range(n + 1)])
    assert abs(value - ref) < tol


if __name__ == "__main__":
    test_clenshaw()
