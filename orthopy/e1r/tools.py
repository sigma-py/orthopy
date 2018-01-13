# -*- coding: utf-8 -*-
#
import matplotlib.pyplot as plt
import numpy

from .orth import tree


def show(*args, **kwargs):
    plot(*args, **kwargs)
    plt.show()
    return


def plot(L):
    xlim = [0.0, +5.0]
    x = numpy.linspace(xlim[0], xlim[1], 500)
    vals = tree(x, L)

    for val in vals:
        plt.plot(x, val)

    # plt.axes().set_aspect('equal')

    plt.xlim(*xlim)
    # plt.ylim(-2, +2)
    plt.tick_params(
        axis='both',
        which='both',
        bottom='off',
        top='off',
        left='off',
        right='off',
        labelbottom='off',
        labelleft='off'
        )
    plt.grid()
    return
