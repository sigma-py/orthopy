import itertools

import numpy

from .orth import Iterator


def show(*args, **kwargs):
    import matplotlib.pyplot as plt

    plot(*args, **kwargs)
    plt.show()


def plot(L):
    import matplotlib.pyplot as plt

    xlim = [-2.0, +2.0]
    x = numpy.linspace(xlim[0], xlim[1], 500)

    for val in itertools.islice(Iterator(x, "normal"), L + 1):
        plt.plot(x, val)

    plt.xlim(*xlim)
    # plt.ylim(-2, +2)
    plt.tick_params(
        axis="both",
        which="both",
        bottom="off",
        top="off",
        left="off",
        right="off",
        labelbottom="off",
        labelleft="off",
    )
    plt.grid()
