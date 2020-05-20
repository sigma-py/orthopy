import itertools


def get_nth(iterator, n):
    return next(itertools.islice(iterator, n, None))
