# -*- coding: utf-8 -*-
"""a function used to compute the loss."""

import numpy as np




def compute_loss(y, tx, w, mse=True):
    """Calculate the loss using either MSE or MAE.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    e = y - tx@w
    if not mse:
        return 1/(y.shape[0]) * np.sum(np.abs(e))
    return 1/(2*y.shape[0]) * e@e