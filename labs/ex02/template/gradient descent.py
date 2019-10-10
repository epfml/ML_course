# -*- coding: utf-8 -*-
"""Gradient Descent"""

def compute_gradient(y, tx, w):
    """Compute the gradient."""
    # ***************************************************
    n = np.shape(tx)[0]
    error = y - np.dot(tx,w)
    return - np.dot(tx.T,error)/n


def gradient_descent(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm with mse."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # ***************************************************
        grad = compute_gradient(y, tx, w)
        loss = compute_loss(y, tx, w, method="mse")
        w = w - gamma*grad
        # ***************************************************
        # store w and loss
        ws.append(w)
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses, ws