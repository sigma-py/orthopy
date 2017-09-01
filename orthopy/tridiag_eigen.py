# -*- coding: utf-8 -*-
#
import mpmath


def tridiag_eigen(in_d, in_e, m=None):
    '''This routine computes the eigenvalues and first m components of the
    eigenvectors of the symmetric tridiagonal matrix [e, d, e].

    This method returns the eigenvalues and the first m components of the
    eigenvectors. If m is not specified, the entire eigenvectors are returned.

    This is an adaptation of mpmath's tridiag_eigen
    <https://github.com/fredrik-johansson/mpmath/blob/master/mpmath/matrices/eigen_symmetric.py#L377>
    to be used until the latter is part of the public interface of mpmath (see
    <https://github.com/fredrik-johansson/mpmath/issues/366>).
    '''
    d = in_d.copy()
    e = in_e.copy()

    n = len(d)

    if m is None:
        m = n

    z = mpmath.zeros(m, n)
    for i in range(m):
        z[i, i] = 1

    ctx = d.ctx

    e[n-1] = 0
    iterlim = 2 * ctx.dps

    for l in range(n):
        j = 0
        while 1:
            m = l
            while 1:
                # look for a small subdiagonal element
                if m + 1 == n:
                    break
                if abs(e[m]) <= ctx.eps * (abs(d[m]) + abs(d[m + 1])):
                    break
                m = m + 1
            if m == l:
                break

            assert j < iterlim, 'no convergence to an eigenvalue'

            j += 1

            # form shift

            p = d[l]
            g = (d[l + 1] - p) / (2 * e[l])
            r = ctx.hypot(g, 1)

            if g < 0:
                s = g - r
            else:
                s = g + r

            g = d[m] - p + e[l] / s

            s, c, p = 1, 1, 0

            for i in range(m - 1, l - 1, -1):
                f = s * e[i]
                b = c * e[i]
                # this here is a slight improvement also used in gaussq.f or
                # acm algorithm 726.
                if abs(f) > abs(g):
                    c = g / f
                    r = ctx.hypot(c, 1)
                    e[i + 1] = f * r
                    s = 1 / r
                    c = c * s
                else:
                    s = f / g
                    r = ctx.hypot(s, 1)
                    e[i + 1] = g * r
                    c = 1 / r
                    s = s * c
                g = d[i + 1] - p
                r = (d[i] - g) * s + 2 * c * b
                p = s * r
                d[i + 1] = g + p
                g = c * r - b

                # calculate eigenvectors
                for w in range(z.rows):
                    f = z[w, i+1]
                    z[w, i+1] = s * z[w, i] + c * f
                    z[w, i] = c * z[w, i] - s * f

            d[l] = d[l] - p
            e[l] = g
            e[m] = 0

    for ii in range(1, n):
        # sort eigenvalues and eigenvectors (bubble-sort)
        i = ii - 1
        k = i
        p = d[i]
        for j in range(ii, n):
            if d[j] >= p:
                continue
            k = j
            p = d[k]
        if k == i:
            continue
        d[k] = d[i]
        d[i] = p

        for w in range(z.rows):
            p = z[w, i]
            z[w, i] = z[w, k]
            z[w, k] = p
    return d, z
