import numpy as np
import sympy


def full_like(x, val):
    if isinstance(x, np.ndarray):
        return np.full_like(x, val)

    # assume x is just a float or int or sympy.Poly
    return x * 0 + val


class Eval1D:
    def __init__(self, x, rc):
        self.rc = rc
        self.x = x
        self.k = 0
        self.last = [None, None]

    def __iter__(self):
        return self

    def __next__(self):
        if self.k == 0:
            out = full_like(self.x, self.rc.p0)
        else:
            a, b, c = self.rc[self.k - 1]
            out = self.last[0] * (self.x * a - b)
            if self.k > 1:
                out -= self.last[1] * c

        self.last[1] = self.last[0]
        self.last[0] = out
        self.k += 1
        return out


class ProductEvalWithDegrees:
    """Evaluates the entire tree of orthogonal polynomials for an n-dimensional product
    domain.

    The computation is organized such that tree returns a list of arrays, L={0, ...,
    dim}, where each level corresponds to the polynomial degree L. Further, each level
    is organized like a discrete (dim-1)-dimensional simplex. Let's demonstrate this for
    3D:

    L = 1:
                   (0, 0, 0)

    L = 2:
                   (1, 0, 0)
              (0, 1, 0) (0, 0, 1)

    L = 3:
                   (2, 0, 0)
              (1, 1, 0) (1, 0, 1)
         (0, 2, 0) (0, 1, 1) (0, 0, 2)

    L = 4:
                   (3, 0, 0)
              (2, 1, 0) (2, 0, 1)
         (1, 2, 0) (1, 1, 1) (1, 0, 2)
    (0, 3, 0) (0, 2, 1) (0, 1, 2) (0, 0, 3)

    The main insight here that makes computation for n dimensions easy is that the next
    level is composed by:

       * Taking the whole previous level and adding +1 to the first entry.
       * Taking the last row of the previous level and adding +1 to the second entry.
       * Taking the last entry of the last row of the previous and adding +1 to the
         third entry.

    In the same manner this can be repeated for `dim` dimensions.
    """

    def __init__(self, rc, int_1, X):
        self.rc = rc

        self.a = None
        self.b = None
        self.c = None
        X = np.asarray(X)
        self.dim = X.shape[0]
        self.p0n = rc.p0 ** self.dim
        self.int_p0 = self.p0n * int_1 ** self.dim
        self.L = 0
        self.X = X
        self.last_values = [None, None]
        self.last_degrees = [None, None]

    def __iter__(self):
        return self

    def __next__(self):
        X = self.X
        dim = X.shape[0]

        if self.L == 0:
            values = np.array([X[0] * 0 + self.p0n])
            degrees = np.array([np.zeros(dim, dtype=int)])
        else:
            aa, bb, cc = self.rc[self.L - 1]
            # cannot just np.append here since numpy will convert to float64
            # https://github.com/numpy/numpy/issues/18189
            self.a = np.array([aa]) if self.a is None else np.append(self.a, aa)
            self.b = np.array([bb]) if self.b is None else np.append(self.b, bb)
            self.c = np.array([cc]) if self.c is None else np.append(self.c, cc)

            a = self.a
            b = self.b
            c = self.c

            values = []
            degrees = []

            mask0 = np.ones(len(self.last_degrees[0]), dtype=bool)
            if self.L > 1:
                mask1 = np.ones(len(self.last_degrees[1]), dtype=bool)

            for i in range(dim):
                lv0 = self.last_values[0][mask0]
                idx0 = self.last_degrees[0][mask0][:, i]

                val = lv0 * (np.multiply.outer(a[idx0], X[i]).T - b[idx0]).T

                if self.L > 1:
                    lv1 = self.last_values[1][mask1]
                    idx1 = self.last_degrees[1][mask1][:, i]
                    yy = idx1 + 1 > 0
                    val[: len(idx1)][yy] -= (lv1[yy].T * c[idx1[yy] + 1]).T

                values.append(val)

                deg = self.last_degrees[0][mask0]
                deg[:, i] += 1
                degrees.append(deg)
                # mask is True for all entries where the first `i` degrees are 0
                mask0 &= self.last_degrees[0][:, i] == 0
                if self.L > 1:
                    mask1 &= self.last_degrees[1][:, i] == 0

            values = np.concatenate(values)
            degrees = np.concatenate(degrees)

        self.last_values[1] = self.last_values[0]
        self.last_values[0] = values

        self.last_degrees[1] = self.last_degrees[0]
        self.last_degrees[0] = degrees
        self.L += 1

        return values, degrees


class ProductEval(ProductEvalWithDegrees):
    """Same as ProductEvalWithDegrees, but next() only returns the values."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __next__(self):
        vals, _ = super().__next__()
        return vals


class Eval135:
    """Evaluates a 1-3-5-tree as seen with associated Legendre polynomials and spherical
    harmonics.

    There are many recurrence relations that can be used to construct the associated
    Legendre polynomials. However, only few are numerically stable. Many implementations
    (including this one) use the classical Legendre recurrence relation with increasing
    L.

    The return value is a list of arrays, where `values[k]` hosts the `2*k+1` values of
    the `k`th level of the tree

                              (0, 0)
                    (-1, 1)   (0, 1)   (1, 1)
          (-2, 2)   (-1, 2)   (0, 2)   (1, 2)   (2, 2)
            ...       ...       ...     ...       ...
    """

    def __init__(self, rc, x, xi=None, symbolic=False):
        self.rc = rc

        self.k = 0
        self.x = x
        # xi[0] == sqrt(1 - x**2) / exp(i*phi)
        # xi[1] == sqrt(1 - x**2) * exp(i*phi)
        if xi is None:
            sqrt = np.vectorize(sympy.sqrt) if symbolic else np.sqrt
            # Such functions aren't always polynomials, see, e.g.,
            # <https://en.wikipedia.org/wiki/Associated_Legendre_polynomials>:
            #
            # > In general, when l and m are integers, the regular solutions are
            # > sometimes called "associated Legendre polynomials", even though they are
            # > not polynomials when m is odd.
            a = sqrt(1 - x ** 2)
            self.xi = [a, a]
        else:
            self.xi = xi

        self.last = [None, None]

    def __iter__(self):
        return self

    def __next__(self):
        if self.k == 0:
            out = np.array([full_like(self.x, self.rc.p0)])
        else:
            z0, z1, c0, c1 = self.rc[self.k]
            out = np.concatenate(
                [
                    [self.last[0][0] * (self.xi[0] * z0)],
                    self.last[0] * np.multiply.outer(c0, self.x),
                    [self.last[0][-1] * (self.xi[1] * z1)],
                ]
            )

            if self.k > 1:
                out[2:-2] -= (self.last[1].T * c1).T

        self.last[1] = self.last[0]
        self.last[0] = out
        self.k += 1
        return out
