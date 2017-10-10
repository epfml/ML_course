# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np

def build_poly(x, degree):
    N = len(x)
    matrix = np.zeros((N,degree+1))
    for i in range(0, N):
        for j in range(0, degree+1):
            matrix[i][j] = x[i]**j
    return matrix
