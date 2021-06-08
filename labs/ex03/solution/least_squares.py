# -*- coding: utf-8 -*-
"""Exercise 3.

Least Squares Solutions
"""

import numpy as np


def least_squares(y, tx):
    """calculate the least squares solution."""
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    return np.linalg.solve(a, b)
