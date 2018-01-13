# -*- coding: utf-8 -*-
#
from __future__ import division

from ..enr2 import tree as enr2_tree


def tree(X, n, symbolic=False):
    assert X.shape[0] == 2, 'X has incorrect shape (X.shape[0] != 2).'
    return enr2_tree(X, n, symbolic=symbolic)
