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
    limit = int(ratio*x.shape[0])
    indices_train = np.random.choice(y.shape[0],limit)
    indices_test = np.delete(np.arange(y.shape[0]),indices_train)
    #shuffle sur les indices --> etre sur que l'on a les memes !!
    
    x_train = x[indices_train]
    x_test = x[indices_test]
    y_train = y[indices_train]
    y_test = y[indices_test]
    return x_train, y_train, x_test, y_test