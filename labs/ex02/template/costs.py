# -*- coding: utf-8 -*-
"""Function used to compute the loss."""

def compute_loss(y, tx, w):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """
    # ***************************************************
    error = y - np.dot(tx,w)
    if method == "mae":
        return np.sum(np.abs(error)) / np.shape(y)[0] / 2
    elif method == "mse":
        return np.inner(error,error) / np.shape(y)[0] / 2 #for MSE
    else:
        raise Exception("spam")
    # ***************************************************