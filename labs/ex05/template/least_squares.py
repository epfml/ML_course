# -*- coding: utf-8 -*-
"""Exercise 3.

Least Square
"""

import numpy as np

def compute_mse(y, tx, w):
    """compute the loss by mse."""
    e = y - tx.dot(w)
    mse = (e**2) / (2 * len(e))
    return mse

def least_squares(y, tx):
    """calculate the least squares solution.
    returns mse, and optimal weights"""
    XX = tx.T.dot(tx)
    B = tx.T.dot(y)
    wstar, _, _, _ = np.linalg.lstsq(XX, B, rcond=None) #Solves Xt*X*W=Xt*Y   
    mse = compute_mse(y, tx, wstar)
    return wstar, mse
