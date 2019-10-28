from helpers import *
import math
from costs import *

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """    Linear regression using gradient descent. Returns the last loss and last ws.    """
    w = initial_w
    for num_iter in range(max_iters):
        grad, e = compute_gradient(y, tx, w)
        loss = compute_loss(y, tx, w)
        w = w - gamma * grad
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
    """    Ridge regression, almost the same as least square but with a with a regularization term    """
    lambda_prime=2*len(y)*lambda_
    a= tx.T.dot(tx) + lambda_prime*np.identity(np.shape(tx)[1]) #tx is N*D, we want a DxD identity
    b= tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss=compute_loss(y, tx, w, type='mae') 
    return w,loss


def calc_loss_log(sig, y):
    variation = 1e-5 #to avoid log(0)
    return ((-y*np.log(sig+variation) - (1-y)*np.log(1-sig+variation)).sum())/len(y)


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    return reg_logistic_regression(y, tx, 0, initial_w, max_iters, gamma)


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    w = initial_w
    threshold = 1e-7
    losses = []
    rate=1 

    for n_iter in range(max_iters):    
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=1, num_batches=1):
            sig = sigmoid(np.dot(tx_batch, w))
            
            gradient = tx_batch.T.dot(sig-y_batch) + lambda_*w         
            w -= (gamma/np.sqrt(n_iter+1)) * gradient
            
            # log info
            if n_iter % 10000 == 0:
                loss = calc_loss_log(sig, y) + (0.5*lambda_)*np.dot(w.T, w)
                losses.append(loss)
                print("Current iteration = {i} loss = {loss}".format(i=n_iter,loss=loss), end="")
                if n_iter != 0 : 
                    rate=np.abs(losses[-2]-losses[-1])
                    print(" rate = " + str(rate))
        if rate<threshold and rate is not math.isnan(rate) : 
            break;
    loss = calc_loss_log(sigmoid(np.dot(tx, w)), y) + (0.5*lambda_)*np.dot(w.T, w)
    print("Best last loss = " + str((loss)) + " after " + str(n_iter) + " iterations ")
    return w, loss 
    
    
    



