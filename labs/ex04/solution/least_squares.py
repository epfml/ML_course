# -*- coding: utf-8 -*-
"""Exercise 3.

Least Square
"""

import numpy as np

### SOLUTION
def compute_mse(e):
    """Calculate the mse for vector e.
    Args:
        e: numpy array of shape (N,), N is the number of samples.

    Returns:
        scalar

    >>> compute_mse(np.array([1.5, -.5]))
    0.625
    """
    return 1 / 2 * np.mean(e**2)


def compute_loss(y, tx, w):
    """Calculate the mse loss.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        w: numpy array of shape(D,)

    Returns:
        Scalar

    >>> compute_loss(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]), np.array([3., 2.1]))
    47.96262500000001
    """
    e = y - tx.dot(w)
    return compute_mse(e)


### TEMPLATE
### END SOLUTION


def least_squares(y, tx):
    """Calculate the least squares solution.
       returns mse, and optimal weights.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar.

    >>> least_squares(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]))
    (array([ 0.21212121, -0.12121212]), 8.666684749742561e-33)
    """
    ### SOLUTION
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    mse = compute_loss(y, tx, w)
    return w, mse
    ### TEMPLATE
    # # ***************************************************
    # # COPY YOUR CODE FROM EX03 HERE
    # # least squares: TODO
    # # returns optimal weights, MSE
    # # ***************************************************
    # raise NotImplementedError
    ### END SOLUTION
