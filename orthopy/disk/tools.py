# -*- coding: utf-8 -*-
#
import matplotlib
import matplotlib.pyplot as plt


def show(*args, **kwargs):
    plot(*args, **kwargs)
    plt.show()
    return


def plot(f, lcar=0.1):
    '''Plot function over a disk.
    '''
    import pygmsh

    geom = pygmsh.built_in.Geometry()
    geom.add_circle(
            [0.0, 0.0, 0.0],
            1.0,
            lcar,
            num_sections=4,
            compound=True
            )
    points, cells, _, _, _ = pygmsh.generate_mesh(geom)

    x = points[:, 0]
    y = points[:, 1]
    triang = matplotlib.tri.Triangulation(x, y, cells['triangle'])

    plt.tripcolor(triang, f(points.T), shading='flat')
    plt.colorbar()

    # # circle outlines
    # X = numpy.column_stack([corners, corners[:, 0]])
    # plt.plot(X[0], X[1], '-k')

    plt.gca().set_aspect('equal')
    plt.axis('off')
    return
