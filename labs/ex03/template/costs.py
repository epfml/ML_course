# -*- coding: utf-8 -*-
"""a function used to compute the loss."""

import numpy as np

def mean_square_error(y, tx, w):
    N = len(y)
    loss = 1/(2*N)*sum([y[n]  - np.dot(tx[n], w)**2 for n in range(N)])
    return loss

def compute_loss(y, tx, w):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """
    mse = mean_square_error(y, tx, w)
    return mse

