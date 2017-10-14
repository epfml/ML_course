# -*- coding: utf-8 -*-
"""Stochastic Gradient Descent"""

def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    
    # number of samples in the batch tx
    B = len(y)
    
    # compute the vector of all the errors
    e = y-tx.dot(w) 

    # compute the stochastic gradient
    stoch_grad = -(1/B)*tx.T.dot(e)
    
    return stoch_grad
    raise NotImplementedError


def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    
    # optimization loop
    for n_iter in range(max_iters):
                
        # pick randomly 'batch_size' samples
        batches = batch_iter(y, tx, batch_size, num_batches=1, shuffle=True)
        
        for samples in batches:

            # read samples
            y = samples[0]
            tx = samples[1]
        
            # compute new parameters
            w = ws[-1] - gamma*compute_stoch_gradient(y, tx, ws[-1])
            
            # get new loss
            loss = compute_mse(y, tx, ws[-1])        
        
        # store w and loss
        ws.append(w)
        losses.append(loss)
        
    print("Gradient Descent({bi}/{ti}): loss MSE={l}, w0={w0}, w1={w1}".format(bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    raise NotImplementedError
    return losses, ws