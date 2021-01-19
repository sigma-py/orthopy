import itertools

import numpy as np

from .main import Eval


def plot(n, *args, **kwargs):
    import dufte
    from matplotlib import pyplot as plt

    plt.style.use(dufte.style)

    x = np.linspace(-2.2, 2.2, 100)
    for k, level in enumerate(itertools.islice(Eval(x, *args, **kwargs), n + 1)):
        plt.plot(x, level, label=f"n={k}")

    plt.grid(axis="x")
    dufte.legend()

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
