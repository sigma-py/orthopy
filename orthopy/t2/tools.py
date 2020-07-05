import numpy

from .main import Eval


def savefig_single(filename, *args, **kwargs):
    import matplotlib.pyplot as plt

    plot_single(*args, **kwargs)
    plt.savefig(filename, transparent=True, bbox_inches="tight")


def show_single(*args, **kwargs):
    import matplotlib.pyplot as plt

    plot_single(*args, **kwargs)
    plt.show()


def plot_single(
    degrees, res=100, scaling="normal", colorbar=True, cmap="viridis", corners=None
):
    """Plot function over a triangle.
    """
    import matplotlib.pyplot as plt
    import meshzoo

    n = sum(degrees)
    r = degrees[0]

    def f(bary):
        for k, level in enumerate(Eval(bary, scaling)):
            if k == n:
                return level[r]

    if corners is None:
        alpha = numpy.pi * numpy.array([7.0 / 6.0, 11.0 / 6.0, 3.0 / 6.0])
        corners = numpy.array([numpy.cos(alpha), numpy.sin(alpha)])

    bary, cells = meshzoo.triangle(res)
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
    plt.title(
        f"Orthogonal polynomial on triangle ([{degrees[0]}, {degrees[1]}], {scaling})"
    )
