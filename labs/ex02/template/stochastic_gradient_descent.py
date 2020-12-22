# -*- coding: utf-8 -*-
"""Stochastic Gradient Descent"""

def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    e = y - tx.dot(w)
    return - 1/len(e) * tx.T.dot(e), e


def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    losses = []
    ws = []
    w = initial_w
    print("Stochastic Gradient Descent: batch_size={bs}, max_iterations={mi}".format(
    bs=batch_size, mi=max_iters))
    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
                #compute gradient and loss and update w
                g, err = compute_stoch_gradient(minibatch_y, minibatch_tx, w)
                loss = calculate_mse(err)
                w = w - gamma * g
                #append ws and losses
                ws.append(w)
                losses.append(loss)
                print("Stochastic Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
                  bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    return losses, ws