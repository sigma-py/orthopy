# -*- coding: utf-8 -*-
#
import matplotlib.tri
import matplotlib.pyplot as plt
import numpy


def show(*args, **kwargs):
    plot(*args, **kwargs)
    plt.show()
    return


def plot(corners, f, n=100):
    '''Plot function over a triangle.
    '''
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
    X = numpy.sum([
        numpy.outer(bary[k], corners[:, k]) for k in range(3)
        ], axis=0)

    # plot the points
    # plt.plot(X[:, 0], X[:, 1], 'xk')

    x = numpy.array(X[:, 0])
    y = numpy.array(X[:, 1])
    z = numpy.array(f(bary), dtype=float)

    triang = matplotlib.tri.Triangulation(x, y)
    plt.tripcolor(triang, z, shading='flat')
    plt.colorbar()

    # triangle outlines
    X = numpy.column_stack([corners, corners[:, 0]])
    plt.plot(X[0], X[1], '-k')

    plt.gca().set_aspect('equal')
    plt.axis('off')
    return
