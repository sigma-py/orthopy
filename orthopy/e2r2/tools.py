# -*- coding: utf-8 -*-
#
import matplotlib.tri
import matplotlib.pyplot as plt
import numpy


def show(*args, **kwargs):
    plot(*args, **kwargs)
    plt.show()
    return


def plot(f, n=100, d=1.0):
    x0 = -d
    x1 = +d
    y0 = -d
    y1 = +d
    x = numpy.linspace(x0, x1, n)
    y = numpy.linspace(y0, y1, n)
    X, Y = numpy.meshgrid(x, y)
    XY = numpy.stack([X, Y])

    z = numpy.array(f(XY), dtype=float)

    triang = matplotlib.tri.Triangulation(X.flatten(), Y.flatten())
    plt.tripcolor(triang, z.flatten(), shading='flat')
    plt.colorbar()

    plt.gca().set_aspect('equal')
    plt.axis('off')
    return
