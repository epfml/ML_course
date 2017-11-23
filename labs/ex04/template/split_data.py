# -*- coding: utf-8 -*-
"""Exercise 3.

Split the dataset based on the given ratio.
"""


import numpy as np


def split_data(x, y, ratio, seed=1):
    """
    split the dataset based on the split ratio. If ratio is 0.8 
    you will have 80% of your data set dedicated to training 
    and the rest dedicated to testing
    """
    
    # create random indices
    np.random.seed(seed)
    index = np.arange(len(x))
    np.random.shuffle(index)

    # separation index
    p = round(ratio * len(y))
    
    # training 
    x_tr = x[index[0:p]]
    y_tr = y[index[0:p]]

    # testing 
    x_te = x[index[p:]]
    y_te = y[index[p:]]
    
    return x_tr, y_tr, x_te, y_te
    
    raise NotImplementedError
