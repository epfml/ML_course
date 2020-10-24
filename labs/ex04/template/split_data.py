# -*- coding: utf-8 -*-
"""Exercise 3.

Split the dataset based on the given ratio.
"""


import numpy as np


def split_data(x, y, ratio, seed=1):
    """split the dataset based on the split ratio."""
    # set seed
    np.random.seed(seed)
    # ***************************************************
    indices = np.random.permutation(len(y))
    
    splits = [int(len(y)*ratio)]    #.cum() if severals
    x_train, x_test = np.split(x[indices], splits)
    y_train, y_test = np.split(y[indices], splits)
    # ***************************************************
    
    return x_train, y_train, x_test, y_test
