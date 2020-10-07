# -*- coding: utf-8 -*-
"""Function used to compute the loss."""
import numpy as np

def compute_mse(y, tx, w):
    """Calculate the mse for vector e."""
    e = y - tx.dot(w)
    return 0.5*np.mean(e**2)


def compute_mae(y, tx, w):
    e = y - tx.dot(w)
    return np.mean(np.abs(e))

def compute_loss(y, tx, w):
    """Calculate the loss.
    You can calculate the loss using mse or mae.
    """
    #MSE
    e = y - tx.dot(w)
    N = len(y)
    loss = (0.5/N) * e.T.dot(e)
    return loss