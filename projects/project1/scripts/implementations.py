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


def calc_loss_log(sig, y):
    variation = 1e-5 #to avoid log(0)
    return ((-y*np.log(sig+variation) - (1-y)*np.log(1-sig+variation)).sum())/len(y)


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    w = initial_w
    losses=[]
    threshold = 1e-5

    for i in range(max_iters):
        sig = sigmoid(np.dot(tx, w))
        gradient = np.dot(tx.T, sig-y)
        w -= gamma*gradient
        
        loss = calc_loss_log(sig, y)
        
        # log info
        if i % 500 == 0:
            print("Current iteration = {i}, loss = {l} ".format(i=i, l=loss), end="")
            if i != 0 : 
                print("rate = " + str(losses[-2]-losses[-1]))
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    print("last loss = " + str(loss) + " after " + str(i) + " iterations ")
    return w, loss



def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    w = initial_w
    threshold = 1e-7
    losses = []
    
    for n_iter in range(max_iters):    
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=4, num_batches=4):
            sig = sigmoid(np.dot(tx_batch, w))
            #sig = sigmoid(np.dot(tx, w))
            gradient = tx_batch.T.dot(sig-y_batch)+lambda_*w
            #gradient = tx.T.dot(sig-y)+lambda_*w
            
            #w -= gamma*gradient
            
            loss = calc_loss_log(sigmoid(np.dot(tx, w)), y) + (0.5*lambda_)*np.dot(w.T, w)
            sig[np.where(sig <= 0.5)] = -1
            sig[np.where(sig > 0.5)] = 1
            w -= gamma*gradient
            
            #loss = calc_loss_log(sig, y) + (0.5*lambda_)*np.dot(w.T, w)
            
            # log info
            if n_iter % 2000 == 0:
                print("Current iteration = {i}, loss = {l} ".format(i=n_iter, l=loss), end="")
                if n_iter != 0 : 
                    print("rate = " + str(losses[-2]-losses[-1]))
            # converge criterion
            losses.append(loss)
            if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
                break
    print("Best last loss = " + str(np.min(losses[-1])) + " after " + str(n_iter) + " iterations ")
    return w, np.min(losses[-1]) #should I return the min or the mean ? 
    
    
    



