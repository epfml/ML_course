# -*- coding: utf-8 -*-
"""Stochastic Gradient Descent"""

def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO: implement stochastic gradient computation.It's same as the gradient descent.
    #--> inutile, utiliser simplement compute_gradient
    # ***************************************************
    raise NotImplementedError


def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    # ***************************************************
    ws = [initial_w]
    grad = 0
    losses = []
    w = initial_w
    for i in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
                grad = compute_gradient(minibatch_y, minibatch_tx,w)
        grad = grad / batch_size    
        w = w - grad * gamma
        ws.append(w)
        losses.append(compute_loss(y, tx, w))
    # ***************************************************
    return losses, ws