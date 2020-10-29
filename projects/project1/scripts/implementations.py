# -*- coding: utf-8 -*-
import numpy as np
from proj1_helpers import *


"""Least squares"""

def least_squares(y, tx):
    """calculate the least squares solution using normal equations.
    returns optimal weights and mse"""
    XX = tx.T.dot(tx)
    B = tx.T.dot(y)
    
    wstar = np.linalg.solve(XX, B) #Solves Xt*X*W=Xt*Y
    
    mse = compute_mse(y, tx, wstar)
    
    return wstar, mse

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """calculate the least squares solution using gradient descent.
    returns optimal weights and mse"""
    ws, losses = gradient_descent(y, tx, initial_w, max_iters, gamma)
    return ws[-1], losses[-1]


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """calculate the least squares solution using stochastic gradient descent.
    returns optimal weights and mse"""
    ws, losses = stochastic_gradient_descent(y, tx, initial_w, 1, max_iters, gamma)
    return ws[max_iters-1], losses[max_iters-1]


"""Ridge Regression"""
def ridge_regression(y, tx, lambda_):
    """implement ridge regression using normal equations.
        return optimal weights and loss"""
    lambda_prime = 2*len(y)*lambda_
    A = tx.T.dot(tx) + (lambda_prime*np.identity(tx.shape[1]))
    B = tx.T.dot(y)
    wstar = np.linalg.solve(A, B)
    
    loss = compute_mse(y, tx, wstar)
    return wstar, loss


"""Logistic Regression"""

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    
    # init parameters
    threshold = 10e-8
    losses = []
    w = initial_w

    # start the logistic regression
    for n_iter in range(max_iters):
        # get loss and update w.
        w, loss = learning_by_gradient_descent(y, tx, w, gamma)
        
        losses.append(loss)
        
        # converge criterion
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
        
        # log info
        if n_iter % 500 == 0:
            print("Current iteration={i}, w={w} loss={l}".format(i=n_iter, w=w, l=loss))
        
    return w, loss



"""Regularized Logistic Regression"""
def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    # init parameters
    threshold = 10e-8
    losses = []
    w = initial_w

    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        w, loss = learning_by_penalized_gradient(y, tx, w, gamma, lambda_)
        # log info
        if iter % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break

    print("best loss={l}".format(l=calculate_loss(y, tx, w)))
    return w, loss