# -*- coding: utf-8 -*-
#
from __future__ import division

from ..ncube import tree as ncube_tree


def tree(n, X, symbolic=False):
    return ncube_tree(n, X, symbolic=symbolic)
