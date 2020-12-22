# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    # ***************************************************
    expand=np.ones((x.shape[0],1))
    for i in range(1,degree+1):
        expand=np.hstack((expand,x.reshape((-1,1))**i))
    # ***************************************************
    return expand