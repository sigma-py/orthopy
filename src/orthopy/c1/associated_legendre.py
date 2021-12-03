from __future__ import annotations

import itertools

import numpy as np
import sympy

from ..helpers import Eval135


def plot(n: int, *args, **kwargs):
    import matplotx
    from matplotlib import pyplot as plt

    plt.style.use(matplotx.styles.dufte)

    x = np.linspace(-1.0, 1.0, 200)
    for k, level in enumerate(itertools.islice(Eval(x, *args, **kwargs), n + 1)):
        # Choose all colors in each level approximately equal, around the reference
        # color.
        ref_color = plt.rcParams["axes.prop_cycle"].by_key()["color"][k]
        ref_rgb = np.array(
            list(int(ref_color[1:][i : i + 2], 16) / 255 for i in (0, 2, 4))
        )
        for l, entry in enumerate(level):
            col = ref_rgb * (1 + (l - k) / (2 * (k + 1)))
            col[col < 0.0] = 0.0
            col[col > 1.0] = 1.0
            plt.plot(x, entry, label=f"n={k}, r={l - k}", color=col)

    plt.grid(axis="x")
    matplotx.line_labels()
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    (scaling,) = args
    plt.title(f'Associated Legendre "polynomials" (scaling={scaling})')
    ax.spines["right"].set_visible(True)
    ax.spines["left"].set_visible(True)
    plt.xlim(-1.0, 1.0)


def show(*args, **kwargs):
    from matplotlib import pyplot as plt

    plot(*args, **kwargs)
    plt.show()


def savefig(filename, *args, **kwargs):
    from matplotlib import pyplot as plt

    plot(*args, **kwargs)
    plt.savefig(filename, transparent=True, bbox_inches="tight")


class Eval:
    """
    Useful references are

    Taweetham Limpanuparb, Josh Milthorpe,
    Associated Legendre Polynomials and Spherical Harmonics Computation for Chemistry
    Applications,
    Proceedings of The 40th Congress on Science and Technology of Thailand;
    2014 Dec 2-4, Khon Kaen, Thailand. P. 233-241.
    <https://arxiv.org/abs/1410.1748>

    and

    Schneider et al.,
    A new Fortran 90 program to compute regular and irregular associated Legendre
    functions,
    Computer Physics Communications,
    Volume 181, Issue 12, December 2010, Pages 2091-2097,
    <https://doi.org/10.1016/j.cpc.2010.08.038>.
    """

    def __init__(self, X, scaling: str, symbolic: str | bool = "auto"):
        cls = {"classical": RCClassical, "normal": RCNormal}[scaling]
        if symbolic == "auto":
            symbolic = np.asarray(X).dtype == sympy.Basic
        rc = cls(symbolic)
        self._eval135 = Eval135(rc, X, symbolic=symbolic)

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._eval135)


class RCClassical:
    def __init__(self, symbolic: bool):
        self.frac = sympy.Rational if symbolic else lambda x, y: x / y
        self.p0 = 1

    def __getitem__(self, L: int):
        z0 = self.frac(1, 2 * L)
        z1 = -(2 * L - 1)
        c0 = [self.frac(2 * L - 1, L - m) for m in range(-L + 1, L)]
        if L == 1:
            c1 = None
        else:
            c1 = [self.frac(L - 1 + m, L - m) for m in range(-L + 2, L - 1)]
        return z0, z1, c0, c1


class RCNormal:
    def __init__(self, symbolic: bool):
        self.sqrt = np.vectorize(sympy.sqrt) if symbolic else np.sqrt
        self.frac = np.vectorize(sympy.Rational) if symbolic else lambda x, y: x / y

        self.p0 = 1 / self.sqrt(2)

    def __getitem__(self, L: int):
        z0 = self.sqrt(self.frac(2 * L + 1, 2 * L))
        z1 = -self.sqrt(self.frac(2 * L + 1, 2 * L))
        #
        m = np.arange(-L + 1, L)
        c0 = self.sqrt(self.frac((2 * L - 1) * (2 * L + 1), (L + m) * (L - m)))
        #
        if L == 1:
            c1 = None
        else:
            m = np.arange(-L + 2, L - 1)
            c1 = self.sqrt(
                self.frac(
                    (L + m - 1) * (L - m - 1) * (2 * L + 1),
                    (2 * L - 3) * (L + m) * (L - m),
                )
            )
        return z0, z1, c0, c1
