# -*- coding: utf-8 -*-
#
import matplotlib
import matplotlib.pyplot as plt


def show(*args, **kwargs):
    plot(*args, **kwargs)
    plt.show()
    return


def plot(f, lcar=1.0e-1):
    '''Plot function over a disk.
    '''
    import pygmsh

    geom = pygmsh.built_in.Geometry()
    geom.add_circle(
            [0.0, 0.0, 0.0],
            1.0,
            lcar,
            num_sections=4,
            compound=True,
            )
    points, cells, _, _, _ = pygmsh.generate_mesh(geom, verbose=True)

    x = points[:, 0]
    y = points[:, 1]
    triang = matplotlib.tri.Triangulation(x, y, cells['triangle'])

    plt.tripcolor(triang, f(points.T), shading='flat')
    plt.colorbar()

    # circle outline
    circle = plt.Circle((0, 0), 1.0, edgecolor='k', fill=False)
    plt.gca().add_artist(circle)

    plt.gca().set_aspect('equal')
    plt.axis('off')
    return
