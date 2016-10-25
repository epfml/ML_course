# -*- coding: utf-8 -*-
"""a function used to compute the loss."""

import numpy as np


def calculate_mse(e):
    """Calculate the mse for vector e."""
    return 1/2*np.mean(e**2)


def calculate_mae(e):
    """Calculate the mae for vector e."""
    return np.mean(np.abs(e))


def compute_loss(y, tx, w):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """
    e = y - tx.dot(w)
    return calculate_mse(e)
    # return calculate_mae(e)


def calculate_nll(y, tx, w):
	"""calculate the negative log likelihood cost."""
	N = y.shape[0]
	xw = np.dot(tx, w)

	loss = np.log(1 + np.exp(xw)) - y*xw


	#for i in range (N):
	#    loss = loss + np.log(1 + np.exp(xw[i])) - y[i]*xw[i]

	return -np.sum(loss, axis=0)
