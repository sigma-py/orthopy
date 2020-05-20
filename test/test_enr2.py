import numpy
import sympy
from sympy import oo

import orthopy


def _integrate3(f, x, y, z):
    return sympy.integrate(
        f * sympy.exp(-(x ** 2 + y ** 2 + z ** 2)),
        (x, -oo, +oo),
        (y, -oo, +oo),
        (z, -oo, +oo),
    )


def test_integral0(n=4):
    xyz = sympy.symbols("x, y, z")
    vals = numpy.concatenate(orthopy.enr2.tree(xyz, n, symbolic=True))

    assert _integrate3(vals[0], *xyz) == sympy.sqrt(sympy.sqrt(sympy.pi)) ** 3
    for val in vals[1:]:
        assert _integrate3(val, *xyz) == 0


def test_orthogonality(n=4):
    x, y, z = sympy.symbols("x, y, z")
    tree = numpy.concatenate(
        orthopy.enr2.tree(numpy.array([x, y, z]), n, symbolic=True)
    )
    vals = tree * numpy.roll(tree, 1, axis=0)

    for val in vals:
        assert _integrate3(val, x, y, z) == 0


def test_normality(n=4):
    xyz = numpy.array(sympy.symbols("x, y, z"))

    for k, level in enumerate(orthopy.enr2.Iterator(xyz, symbolic=True)):
        for val in level:
            assert _integrate3(val ** 2, *xyz) == 1
        if k == n:
            break


if __name__ == "__main__":
    test_normality()
