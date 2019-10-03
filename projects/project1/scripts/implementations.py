from scripts.costs import *
from scripts.proj1_helpers import *
from scripts.helpers import *

#Least squares gradient descent
def compute_gradient(y, tx, w):
    """Compute the gradient."""
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(err)
    return grad, err


#Stochastic gradient descent
def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(err)
    return grad, err

def stochastic_gradient_descent(
        y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    
    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=batch_size, num_batches=1):
            # compute a stochastic gradient and loss
            grad, _ = compute_stoch_gradient(y_batch, tx_batch, w)
            # update w through the stochastic gradient update
            w = w - gamma * grad
            # calculate loss
            loss = compute_loss(y, tx, w)
            # store w and loss
            ws.append(w)
            losses.append(loss)

    """        print("SGD({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))"""
    return losses, w


#########################################33

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """
    Linear regression using gradient descent. Returns the array of losses and the array of weights ws.
    """
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        grad, err = compute_gradient(y, tx, w)
        loss = calculate_mse(err)
        w = w - gamma * grad
        ws.append(w)
        losses.append(loss)
    """        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))"""
    return losses, ws


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """
    TODO
    """
    return null

#Least square normal equation
def least_squares(y, tx): 
    """Least squares regression using normal equations  """
    a=tx.T.dot(tx)
    b=tx.T.dot(y)
    w = np.linalg.solve(a, b) 
    loss=compute_loss(y, tx, w) 
    return loss, w


def ridge_regression(y, tx, lambda_):
    """
    TODO
    """
    return null


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    TODO
    """
    raise NotImplementedError
    return null

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    TODO
    """
    raise NotImplementedError
    return null
