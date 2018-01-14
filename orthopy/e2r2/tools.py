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
    x = numpy.linspace(-d, +d, n)
    y = numpy.linspace(-d, +d, n)
    X, Y = numpy.meshgrid(x, y)
    XY = numpy.stack([X, Y])

    z = numpy.array(f(XY), dtype=float)

    triang = matplotlib.tri.Triangulation(X.flatten(), Y.flatten())
    plt.tripcolor(triang, z.flatten(), shading='flat')
    plt.colorbar()

    # Choose a diverging colormap such that the zeros are clearly
    # distinguishable.
    plt.set_cmap('coolwarm')
    # Make sure the color map limits are symmetric around 0.
    clim = plt.gci().get_clim()
    mx = max(abs(clim[0]), abs(clim[1]))
    plt.clim(-mx, mx)

    plt.gca().set_aspect('equal')
    plt.axis('off')
    return
