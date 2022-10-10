# -*- coding: utf-8 -*-
"""Problem Sheet 2.

Gradient Descent
"""
from costs import *


def compute_gradient(y, tx, w):
    """Computes the gradient at w.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2, ). The vector of model parameters.

    Returns:
        An array of shape (2, ) (same shape as w), containing the gradient of the loss at w.
    """
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO: compute gradient vector
    return -1/len(y)*tx.T@(y-tx@w)
    # ***************************************************
    raise NotImplementedError


def gradient_descent(y, tx, initial_w, max_iters, gamma):
    """The Gradient Descent (GD) algorithm.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        initial_w: shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of GD
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (2, ), for each iteration of GD
    """
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
       # ***************************************************
        # INSERT YOUR CODE HERE
        # TODO: compute gradient and loss
        grad=compute_gradient(y,tx,w)
        loss=compute_loss(y,tx,w)
        # ***************************************************

        # ***************************************************
        # INSERT YOUR CODE HERE
        # TODO: update w by gradient
        w=w-gamma*grad
        # ***************************************************
        
        # store w and loss
        ws.append(w)
        losses.append(loss)

    return losses, ws
