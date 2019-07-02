import numpy

from .orth import tree


def show(*args, **kwargs):
    import matplotlib.pyplot as plt

    plot(*args, **kwargs)
    plt.show()
    return


def plot(L):
    import matplotlib.pyplot as plt

    xlim = [0.0, +5.0]
    x = numpy.linspace(xlim[0], xlim[1], 500)
    vals = tree(x, L)

    for val in vals:
        plt.plot(x, val)

    # plt.axes().set_aspect('equal')

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
    return
