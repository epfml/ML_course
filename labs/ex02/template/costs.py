# -*- coding: utf-8 -*-
"""a function used to compute the loss."""

import numpy as np


def compute_loss(y, tx, w, loss_function='MSE'):
    """Calculate the loss using either MSE or MAE.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO: compute loss by MSE
    N = y.shape[0]
    error = y - np.dot(tx, w)
    if loss_function == 'MSE':
        loss = (1/(2*N))*np.dot(error.transpose(), error)
    elif loss_function == 'MAE':
        loss = (1/N)*np.sum(np.abs(error))
    return loss
    # ***************************************************
