# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""

    poly = np.vander(x, degree+1, increasing=True)
    
    return poly
    raise NotImplementedError
