# -*- coding: utf-8 -*-
"""A function to compute the cost."""


def compute_mse(y, tx, beta):
    """compute the loss by mse."""
    e = y - tx.dot(beta)
    mse = e.dot(e) / (2 * len(e))
    return mse
