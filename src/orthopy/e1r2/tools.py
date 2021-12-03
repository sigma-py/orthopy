import itertools

import numpy as np

from .main import Eval


def plot(n, *args, **kwargs):
    import matplotx
    from matplotlib import pyplot as plt

    plt.style.use(matplotx.styles.dufte)

    x = np.linspace(-2.2, 2.2, 100)
    for k, level in enumerate(itertools.islice(Eval(x, *args, **kwargs), n + 1)):
        plt.plot(x, level, label=f"n={k}")

    plt.grid(axis="x")
    matplotx.line_labels()

    variant, scaling = args
    plt.title(f"Hermite polynomials ({variant}, scaling={scaling})")


def show(*args, **kwargs):
    from matplotlib import pyplot as plt

    plot(*args, **kwargs)
    plt.show()


def savefig(filename, *args, **kwargs):
    from matplotlib import pyplot as plt

    plot(*args, **kwargs)
    plt.savefig(filename, transparent=True, bbox_inches="tight")
