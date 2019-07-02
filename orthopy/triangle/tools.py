from __future__ import division

import numpy


def show(*args, **kwargs):
    import matplotlib.pyplot as plt

    plot(*args, **kwargs)
    plt.show()
    return


def plot(corners, f, n=100):
    """Plot function over a triangle.
    """
    import matplotlib.tri
    import matplotlib.pyplot as plt

    # discretization points
    def partition(boxes, balls):
        # <https://stackoverflow.com/a/36748940/353337>
        def rec(boxes, balls, parent=tuple()):
            if boxes > 1:
                for i in range(balls + 1):
                    for x in rec(boxes - 1, i, parent + (balls - i,)):
                        yield x
            else:
                yield parent + (balls,)

        return list(rec(boxes, balls))

    bary = numpy.array(partition(3, n)).T / n
    X = numpy.sum([numpy.outer(bary[k], corners[:, k]) for k in range(3)], axis=0).T

    # plot the points
    # plt.plot(X[0], X[1], 'xk')

    x = numpy.array(X[0])
    y = numpy.array(X[1])
    z = numpy.array(f(bary), dtype=float)

    triang = matplotlib.tri.Triangulation(x, y)
    plt.tripcolor(triang, z, shading="flat")
    plt.colorbar()

    # Choose a diverging colormap such that the zeros are clearly
    # distinguishable.
    plt.set_cmap("coolwarm")
    # Make sure the color map limits are symmetric around 0.
    clim = plt.gci().get_clim()
    mx = max(abs(clim[0]), abs(clim[1]))
    plt.clim(-mx, mx)

    # triangle outlines
    X = numpy.column_stack([corners, corners[:, 0]])
    plt.plot(X[0], X[1], "-k")

    plt.gca().set_aspect("equal")
    plt.axis("off")
    return
