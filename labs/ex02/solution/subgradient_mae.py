import numpy as np

def compute_subgradient_mae(y, tx, w):
    """Compute the gradient."""
    err = y - tx.dot(w)
    grad = -tx.T.dot(np.sign(err)) / len(err)
    return grad, err