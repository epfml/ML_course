# -*- coding: utf-8 -*-
"""Exercise 3.

Ridge Regression
"""

import numpy as np


def ridge_regression(y, tX, lambda_):
    """implement ridge regression."""
    # ***************************************************
    n = np.shape(tX)[0]
    w_optimal = []
    if tX.ndim > 1:
        d = np.shape(tX)[1]
        w_optimal = np.linalg.solve((np.dot(tX.T,tX) + lambda_*n*2*np.eye(d)) ,np.dot(tX.T,y))
    else:
        d = 1
        w_optimal = np.dot(np.linalg.inv(np.dot(tX.T,tX) + lambda_*n*2*np.eye(d)),np.dot(tX.T,y))
    error = y - np.dot(tX,w_optimal)
    mse =  np.inner(error,error) / n / 2
    # ***************************************************
    return w_optimal, mse