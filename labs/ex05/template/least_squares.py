# -*- coding: utf-8 -*-
"""Exercise 3.

Least Square
"""

import numpy as np


def least_squares(y, tX):
    """calculate the least squares."""
    # ***************************************************
    w_optimal = np.linalg.solve(np.dot(tX.T,tX),np.dot(tX.T,y))
    error = y - np.dot(tX,w_optimal)
    mse =  np.inner(error,error) / np.shape(y)[0] / 2
    return w_optimal, mse
