# -*- coding: utf-8 -*-
"""Function used to compute the loss."""

def compute_loss(y, tx, w):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """
    #MSE
    e = y - tx.dot(w)
    N = len(y)
    loss = (0.5/N) * e.T.dot(e)
    return loss