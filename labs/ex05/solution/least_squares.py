# -*- coding: utf-8 -*-
"""Exercise 3.

Least Square
"""

import numpy as np


def least_squares(y, tx):
    """calculate the least squares."""
    ### SOLUTION
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    return np.linalg.solve(a, b)
    ### TEMPLATE
    # # ***************************************************
    # # INSERT YOUR CODE HERE
    # # least squares: TODO
    # # returns mse, and optimal weights
    # # ***************************************************
    # raise NotImplementedError
    ### END SOLUTION
