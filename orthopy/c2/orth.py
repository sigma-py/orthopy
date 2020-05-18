from ..cn import tree as cn_tree


def tree(X, n, symbolic=False):
    assert X.shape[0] == 2, "X has incorrect shape (X.shape[0] != 2)."
    return cn_tree(X, n, symbolic=symbolic)
