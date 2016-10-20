# -*- coding: utf-8 -*-
"""Problem Sheet 2.

Grid Search
"""

import numpy as np

def compute_mse(y, tx, w):
    N = len(y)
    e = y - np.dot(tx, w)
    loss = 1/(2*N) * np.dot(e.T, e)

    return loss

def compute_loss(y, tx, w):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """
    mse = compute_mse(y, tx, w)
    return mse

def generate_w(num_intervals):
    """Generate a grid of values for w0 and w1."""
    w0 = np.linspace(-100, 200, num_intervals)
    w1 = np.linspace(-150, 150, num_intervals)
    return w0, w1


def get_best_parameters(w0, w1, losses):
    """Get the best w from the result of grid search."""
    min_row, min_col = np.unravel_index(np.argmin(losses), losses.shape)
    weights = [w0[min_row], w1[min_col]]
    loss = losses[min_row, min_col]
    return loss, weights


def grid_search(y, tx, w0, w1):
    """Algorithm for grid search."""
    losses = [[compute_loss(y, tx, (i,j)) for j in w1] for i in w0 ]
    return np.array(losses)