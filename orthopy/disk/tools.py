def show(*args, **kwargs):
    import matplotlib.pyplot as plt

    plot(*args, **kwargs)
    plt.show()
    return


def plot(f, lcar=1.0e-1):
    """Plot function over a disk.
    """
    import matplotlib
    import matplotlib.pyplot as plt
    import dmsh

    geo = dmsh.Circle([0.0, 0.0], 1.0)
    points, cells = dmsh.generate(geo, 0.1)

    x = points[:, 0]
    y = points[:, 1]
    triang = matplotlib.tri.Triangulation(x, y, cells["triangle"])

    plt.tripcolor(triang, f(points.T), shading="flat")
    plt.colorbar()

    # Choose a diverging colormap such that the zeros are clearly
    # distinguishable.
    plt.set_cmap("coolwarm")
    # Make sure the color map limits are symmetric around 0.
    clim = plt.gci().get_clim()
    mx = max(abs(clim[0]), abs(clim[1]))
    plt.clim(-mx, mx)

    # circle outline
    circle = plt.Circle((0, 0), 1.0, edgecolor="k", fill=False)
    plt.gca().add_artist(circle)

    plt.gca().set_aspect("equal")
    plt.axis("off")
    return
