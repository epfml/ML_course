# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    ### SOLUTION
    poly = np.ones((len(x), 1))
    for deg in range(1, degree + 1):
        poly = np.c_[poly, np.power(x, deg)]
    return poly
    ### TEMPLATE
    # # ***************************************************
    # # INSERT YOUR CODE HERE
    # # polynomial basis function: TODO
    # # this function should return the matrix formed
    # # by applying the polynomial basis to the input data
    # # ***************************************************
    # raise NotImplementedError
    ### END SOLUTION
