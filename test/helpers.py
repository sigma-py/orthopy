import itertools
import math
import operator
import sys
from functools import reduce


def get_nth(iterator, n):
    return next(itertools.islice(iterator, n, None))


def prod(iterable):
    if sys.version < "3.8":
        return reduce(operator.mul, iterable, 1)
    return math.prod(iterable)
