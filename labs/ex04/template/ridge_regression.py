# -*- coding: utf-8 -*-
"""Exercise 3.

Ridge Regression
"""

import numpy as np


def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
        
    # optimal weights 
    w_rr = np.linalg.inv(tx.T.dot(tx) + lambda_*np.identity(tx.shape[1])).dot(tx.T).dot(y)
    
    return w_rr
    raise NotImplementedError
