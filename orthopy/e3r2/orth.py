from ..enr2 import tree as enr2_tree


def tree(X, n, symbolic=False):
    assert X.shape[0] == 3, "X has incorrect shape (X.shape[0] != 3)."
    return enr2_tree(X, n, symbolic=symbolic)
