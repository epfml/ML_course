# -*- coding: utf-8 -*-
"""Least Square"""

import numpy as np
from costs import compute_mse

def least_squares(y, tx):
    """calculate the least squares solution.
    returns mse, and optimal weights"""
    XX = tx.T.dot(tx)
    B = tx.T.dot(y)
    wstar, _, _, _ = np.linalg.lstsq(XX, B, rcond=None) #Solves Xt*X*W=Xt*Y   
    mse = compute_mse(y, tx, wstar)
    return wstar, mse
