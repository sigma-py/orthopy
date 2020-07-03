import itertools

import numpy
import sympy

from ..helpers import Eval135


def tree(n, *args, **kwargs):
    return list(itertools.islice(Eval(*args, **kwargs), n + 1))


def plot(n, *args, **kwargs):
    import dufte
    import matplotlib.pyplot as plt

    plt.style.use(dufte.style)

    x = numpy.linspace(-1.0, 1.0, 200)
    for k, level in enumerate(itertools.islice(Eval(x, *args, **kwargs), n + 1)):
        # Choose all colors in each level approximately equal, around the reference
        # color.
        ref_color = plt.rcParams["axes.prop_cycle"].by_key()["color"][k]
        ref_rgb = numpy.array(
            list(int(ref_color[1:][i : i + 2], 16) / 255 for i in (0, 2, 4))
        )
        for l, entry in enumerate(level):
            col = ref_rgb * (1 + (l - k) / (2 * (k + 1)))
            col[col < 0.0] = 0.0
            col[col > 1.0] = 1.0
            plt.plot(x, entry, label=f"n={k}, r={l - k}", color=col)

    plt.grid(axis="x")
    dufte.legend()
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    (scaling,) = args
    plt.title(f'Associated Legendre "polynomials" (scaling={scaling})')
    ax.spines["right"].set_visible(True)
    ax.spines["left"].set_visible(True)
    plt.xlim(-1.0, 1.0)


def show(*args, **kwargs):
    import matplotlib.pyplot as plt

    plot(*args, **kwargs)
    plt.show()


def savefig(filename, *args, **kwargs):
    import matplotlib.pyplot as plt

    plot(*args, **kwargs)
    plt.savefig(filename, transparent=True, bbox_inches="tight")


class Eval(Eval135):
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

    def __init__(self, X, scaling, symbolic=False):
        cls = {"classical": RCClassical, "normal": RCNormal}[scaling]
        rc = cls(symbolic)
        super().__init__(rc, X, symbolic=symbolic)


class RCClassical:
    def __init__(self, symbolic):
        self.frac = sympy.Rational if symbolic else lambda x, y: x / y
        self.p0 = 1

    def __getitem__(self, L):
        z0 = self.frac(1, 2 * L)
        z1 = -(2 * L - 1)
        c0 = [self.frac(2 * L - 1, L - m) for m in range(-L + 1, L)]
        if L == 1:
            c1 = None
        else:
            c1 = [self.frac(L - 1 + m, L - m) for m in range(-L + 2, L - 1)]
        return z0, z1, c0, c1


class RCNormal:
    def __init__(self, symbolic):
        self.sqrt = numpy.vectorize(sympy.sqrt) if symbolic else numpy.sqrt
        self.frac = numpy.vectorize(sympy.Rational) if symbolic else lambda x, y: x / y

        self.p0 = 1 / self.sqrt(2)

    def __getitem__(self, L):
        z0 = self.sqrt(self.frac(2 * L + 1, 2 * L))
        z1 = -self.sqrt(self.frac(2 * L + 1, 2 * L))
        #
        m = numpy.arange(-L + 1, L)
        c0 = self.sqrt(self.frac((2 * L - 1) * (2 * L + 1), (L + m) * (L - m)))
        #
        if L == 1:
            c1 = None
        else:
            m = numpy.arange(-L + 2, L - 1)
            c1 = self.sqrt(
                self.frac(
                    (L + m - 1) * (L - m - 1) * (2 * L + 1),
                    (2 * L - 3) * (L + m) * (L - m),
                )
            )
        return z0, z1, c0, c1
