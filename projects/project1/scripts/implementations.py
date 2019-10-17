from costs import *
from proj1_helpers import *
from helpers import *


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """
    Linear regression using gradient descent. Returns the last loss and last ws.
    """
    w = initial_w
    for num_iter in range(max_iters):
        grad, e = compute_gradient(y, tx, w)
        loss = compute_loss(y, tx, w)
        w = w - gamma * grad
        print(w, loss)
    return w, loss


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Stochastic gradient descent."""
    w = initial_w
    batch_size=1 #Possible to change this value, fixed from project description
    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=batch_size, num_batches=1):
            grad, e = compute_gradient(y_batch, tx_batch, w)
            w = w - gamma * grad
            loss = compute_loss(y, tx, w)
            #print(w, loss)
    return w, loss

#Least square normal equation
def least_squares(y, tx): 
    """Least squares regression using normal equations  """
    a=tx.T.dot(tx)
    b=tx.T.dot(y)
    w = np.linalg.solve(a, b) 
    loss=compute_loss(y, tx, w) 
    return w, loss


def ridge_regression(y, tx, lambda_):
    """
    Ridge regression, almost the same as least square but with a with a regularization term
    """
    lambda_prime=2*len(y)*lambda_
    a=tx.T.dot(tx)+lambda_prime*np.identity(np.shape(tx)[1]) #tx is N*D, we want a DxD identity
    b=tx.T.dot(y)
    w = np.linalg.solve(a, b) 
    loss=compute_loss(y, tx, w) 
    return w,loss

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    TODO
    """
    raise NotImplementedError

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    TODO
    """
    raise NotImplementedError
