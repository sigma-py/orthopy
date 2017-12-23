# -*- coding: utf-8 -*-
#
from __future__ import print_function

from .__about__ import (
    __author__,
    __email__,
    __copyright__,
    __credits__,
    __license__,
    __version__,
    __maintainer__,
    __status__,
    )

from . import e1r2
from . import e2r2
from . import enr2
from . import hexahedron
from . import line
from . import ncube
from . import quadrilateral
from . import sphere
from . import triangle

try:
    import pipdate
except ImportError:
    pass
else:
    if pipdate.needs_checking(__name__):
        print(pipdate.check(__name__, __version__))
