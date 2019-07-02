# -*- coding: utf-8 -*-
#
from __future__ import print_function

from . import (
    disk,
    e1r,
    e1r2,
    e2r2,
    enr2,
    hexahedron,
    line_segment,
    ncube,
    quadrilateral,
    sphere,
    triangle,
)
from .__about__ import (
    __author__,
    __copyright__,
    __email__,
    __license__,
    __status__,
    __version__,
)

__all__ = [
    "__author__",
    "__email__",
    "__copyright__",
    "__license__",
    "__version__",
    "__status__",
    "disk",
    "e1r",
    "e1r2",
    "e2r2",
    "enr2",
    "hexahedron",
    "line_segment",
    "ncube",
    "quadrilateral",
    "sphere",
    "triangle",
]

try:
    import pipdate
except ImportError:
    pass
else:
    if pipdate.needs_checking(__name__):
        print(pipdate.check(__name__, __version__), end="")
