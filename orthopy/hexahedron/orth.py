# -*- coding: utf-8 -*-
#
from __future__ import division

from ..ncube import tree as ncube_tree


def tree(n, X, symbolic=False):
    assert X.shape[0] == 3, 'X has incorrect shape (X.shape[0] != 3).'
    return ncube_tree(n, X, symbolic=symbolic)
