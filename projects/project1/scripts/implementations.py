from scripts.costs import *
from scripts.proj1_helpers import *
from scripts.helpers import *

#Helper functions : (Maybe we should put them in another file)

def compute_gradient(y, tx, w):
    """Compute the gradient."""
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(err)
    return grad, err

def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(err)
    return grad, err

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]

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
    """Stochastic gradient descent."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    
    batch_size=1 #Possible to change this value,m
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
    Ridge regression, almost the same as least square but with a with a regularization term
    """
    lambda_prime=2*len(y)*lambda_
    a=tx.T.dot(tx)+lambda_prime*np.identity(n.shape(tx)[1]) #tx is N*D, we want a DxD identity
    b=tx.T.dot(y)
    w = np.linalg.solve(a, b) 
    loss=compute_loss(y, tx, w) 
    return loss, w

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
