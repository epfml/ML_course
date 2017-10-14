# -*- coding: utf-8 -*-
"""Functions used to compute the loss."""

def compute_mse(y, tx, w):
    """Calculate the loss using mse.
    """
   
    # number of samples
    N = len(y)

    # compute the error
    e = y-tx.dot(w)
    
    # when loss is the MSE
    mse = (1/(2*N))*e.dot(e)
    
    return mse

def compute_mae(y, tx, w):
    """Calculate the loss using mae.
    """
   
    # number of samples
    N = len(y)
    
    # compute the error
    e = y-tx.dot(w)
    
    # when loss is the MAE
    mae = abs(e).mean()
    
    return mae