# -*- coding: utf-8 -*-
#
from __future__ import division

from ..ncube import tree as ncube_tree


def tree(X, n, symbolic=False):
    assert X.shape[0] == 2, "X has incorrect shape (X.shape[0] != 2)."
    return ncube_tree(X, n, symbolic=symbolic)
