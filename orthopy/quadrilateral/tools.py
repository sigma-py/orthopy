# -*- coding: utf-8 -*-
#
import matplotlib.tri
import matplotlib.pyplot as plt
import numpy


def show(*args, **kwargs):
    plot(*args, **kwargs)
    plt.show()
    return


def plot(f, n=100):
    '''Plot function over the standard quadrilateral.
    '''
    x = numpy.linspace(-1, +1, n)
    y = numpy.linspace(-1, +1, n)
    X, Y = numpy.meshgrid(x, y)
    XY = numpy.stack([X, Y])

    z = numpy.array(f(XY), dtype=float)

    triang = matplotlib.tri.Triangulation(X.flatten(), Y.flatten())
    plt.tripcolor(triang, z.flatten(), shading='flat')
    plt.colorbar()

    # quad outlines
    X = numpy.array([[-1, 1, 1, -1, -1], [-1, -1, 1, 1, -1]])
    plt.plot(X[0], X[1], '-k')

    plt.gca().set_aspect('equal')
    plt.axis('off')
    return
