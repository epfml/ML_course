# -*- coding: utf-8 -*-
"""Exercise 2.

Grid Search
"""

import numpy as np
from costs import compute_loss


def generate_w(num_intervals):
    """Generate a grid of values for w0 and w1."""
    w0 = np.linspace(-100, 200, num_intervals)
    w1 = np.linspace(-150, 150, num_intervals)
    return w0, w1


def get_best_parameters(w0, w1, losses):
    """Get the best w from the result of grid search."""
    min_row, min_col = np.unravel_index(np.argmin(losses), losses.shape)
    return losses[min_row, min_col], w0[min_row], w1[min_col]


### SOLUTION
def grid_search(y, tx, w0, w1):
    """Algorithm for grid search."""
    losses = np.zeros((len(w0), len(w1)))
    # compute loss for each combination of w0 and w1.
    for ind_row, row in enumerate(w0):
        for ind_col, col in enumerate(w1):
            w = np.array([row, col])
            losses[ind_row, ind_col] = compute_loss(y, tx, w)
    return losses


### TEMPLATE
# # ***************************************************
# # INSERT YOUR CODE HERE
# # TODO: Paste your implementation of grid_search
# #       here when it is done.
# # ***************************************************
### END SOLUTION
