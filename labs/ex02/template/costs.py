# -*- coding: utf-8 -*-
"""a function used to compute the loss."""

import numpy as np


def compute_loss(y, tx, w):
    """Calculate the loss using either MSE or MAE.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO: compute loss by MSE
    # ***************************************************
    #Using the MSE formula
    #error = y - tx.dot(w)
    #loss = 1/(2*y.shape[0]) * np.sum(error**2)

    #Using the MAE formula
    error = y - tx.dot(w)
    loss = 1/y.shape[0] * np.sum(np.abs(error))
    return loss
