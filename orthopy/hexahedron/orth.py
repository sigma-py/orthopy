from ..ncube import tree as ncube_tree


def tree(X, n, symbolic=False):
    assert X.shape[0] == 3, "X has incorrect shape (X.shape[0] != 3)."
    return ncube_tree(X, n, symbolic=symbolic)
