# -*- coding: utf-8 -*-
#
import numpy


def show(*args, **kwargs):
    plot(*args, **kwargs)
    plt.show()
    return


def plot(f, n=100):
    '''Plot function over the standard quadrilateral.
    '''
    import matplotlib.tri
    import matplotlib.pyplot as plt

    x = numpy.linspace(-1, +1, n)
    y = numpy.linspace(-1, +1, n)
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

    # quad outlines
    X = numpy.array([[-1, 1, 1, -1, -1], [-1, -1, 1, 1, -1]])
    plt.plot(X[0], X[1], '-k')

    plt.gca().set_aspect('equal')
    plt.axis('off')
    return
