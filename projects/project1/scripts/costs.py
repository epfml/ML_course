# -*- coding: utf-8 -*-
"""a function used to compute the loss."""

import numpy as np
from helpers import *

def calculate_mse(e):
    """Calculate the mse for vector e."""
    return 1/2*np.mean(e**2)


def calculate_mae(e):
    """Calculate the mae for vector e."""
    return np.mean(np.abs(e))


def compute_loss(y, tx, w, type='mse'):
    """Calculate the loss.
    You can calculate the loss using mse or mae.
    By default the loss is mse. Raise exception if type is not mse nor mae
    """
    if (type is not 'mae' and type is not 'mse'):
       raise ValueError("type is not mse nor mae")
    e = y - tx.dot(w)
    if type is 'mse':
        return calculate_mse(e)
    else :
        return calculate_mae(e)
