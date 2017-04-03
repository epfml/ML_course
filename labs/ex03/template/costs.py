# -*- coding: utf-8 -*-
"""a function used to compute the loss."""

import numpy as np

def compute_mse(y, tx, w):
    N = len(y)
    e = y - np.dot(tx, w)
    loss = 1/(2*N) * np.dot(e.T, e)

    return loss

def compute_loss(y, tx, w):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """
    mse = compute_mse(y, tx, w)
    return mse

