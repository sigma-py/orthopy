import itertools

import numpy


def tree(n, *args, **kwargs):
    return list(itertools.islice(Eval(*args, **kwargs), n + 1))


class Eval:
    """
    Similar to regular Zernike, but a lot simpler. Can probably be generalized to
    n-ball.
    """

    def __init__(self, X, scaling, symbolic=False):
        self.X = X
        self.L = 0
        self.last = [None, None]
        self.p0 = 1

    def __iter__(self):
        return self

    def __next__(self):
        if self.L == 0:
            out = numpy.array([0 * self.X[0] + self.p0])
        else:
            shape = list(self.last[0].shape)
            shape[0] += 1
            out = numpy.zeros(shape, dtype=self.last[0].dtype)

            last_X = self.last[0] * self.X[0]
            last_Y = self.last[0] * self.X[1]
            last_Y = last_Y[::-1]

            # The minus sign could go onto the other last_Y, too.
            out[1:] += last_X + last_Y
            out[:-1] += last_X - last_Y
            if self.L > 1:
                out[1:-1] -= self.last[1]

        self.last[1] = self.last[0]
        self.last[0] = out
        self.L += 1
        return out
