# -*- coding: utf-8 -*-
#
from __future__ import division

from ..line import recurrence_coefficients, tree as line_tree


def tree(n, X, symbolic=False):
    args = recurrence_coefficients.hermite(
            n, standardization='normal', symbolic=symbolic
            )
    return line_tree(X, *args)
