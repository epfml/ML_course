# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""
import numpy as np
def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    poly = x
    #poly = np.ones(x.shape)
    for deg in range(2, degree+1):
        poly = np.c_[poly, np.power(x, deg)]
    one=np.ones([poly.shape[0],1])
    return np.c_[one,poly]


