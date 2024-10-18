# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # polynomial basis function: TODO
    # this function should return the matrix formed
    # by applying the polynomial basis to the input data
    # ***************************************************
    
    #We can also use np.vander(x, degree+1) to get the same result
    
    poly = np.zeros((len(x), degree+1))
    for i in range(degree+1):
        poly[:, i] = x**i
    return poly
