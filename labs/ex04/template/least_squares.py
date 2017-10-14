# -*- coding: utf-8 -*-
"""Exercise 3.

Least Square
"""

import numpy as np


def least_squares(y, tx):
    """calculate the least squares."""

    # optimal weights 
    w_ls = np.linalg.inv(tx.T.dot(tx)).dot(tx.T).dot(y)
    
    return w_ls
    raise NotImplementedError
