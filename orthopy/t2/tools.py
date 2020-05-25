import numpy


def show(*args, **kwargs):
    import matplotlib.pyplot as plt

    plot(*args, **kwargs)
    plt.show()


def plot(corners, f, n=100, colorbar=True, cmap="viridis"):
    """Plot function over a triangle.
    """
    import matplotlib.pyplot as plt
    import meshzoo

    bary, cells = meshzoo.triangle(n)
    x, y = numpy.dot(corners, bary)
    z = numpy.array(f(bary), dtype=float)

    plt.tripcolor(x, y, cells, z, shading="flat")

    if colorbar:
        plt.colorbar()
    # Choose a diverging colormap such that the zeros are clearly distinguishable.
    plt.set_cmap(cmap)
    # Make sure the color map limits are symmetric around 0.
    clim = plt.gci().get_clim()
    mx = max(abs(clim[0]), abs(clim[1]))
    plt.clim(-mx, mx)

    # triangle outlines
    X = numpy.column_stack([corners, corners[:, 0]])
    plt.plot(X[0], X[1], "-k")

    plt.gca().set_aspect("equal")
    plt.axis("off")
