# -*- coding: utf-8 -*-
"""Exercise 3.

Least Square
"""

import numpy as np


def least_squares(y, tx):
    """calculate the least squares."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # least squares: TODO
    # returns mse, and optimal weights
    # ***************************************************
    w = np.linalg.solve(tx.T.dot(tx), tx.T.dot(y))
    error = y - tx.dot(w)
    mse = 1/(2*len(y)) * np.sum(error**2)
    return w, mse
