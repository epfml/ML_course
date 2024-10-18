import numpy as np
from utils import batch_iter

# compute the mean squared error of a model
def MSE(y, tx, w):
    N = y.shape[0]
    e = y - (tx @ w)
    loss = 1/(2*N) * np.sum(e**2, 0)
    return loss

# compute the gradient of the MSE loss function
def compute_gradient_MSE(y, tx, w):
    N = y.shape[0]
    e = y - (tx @ w)
    grad = -1/N * (tx.T @ e)
    return grad

# compute the regularized MSE loss, return both the non-regularized and the regularized loss
def MSE_regularized(y, tx, w, lambda_):
    loss = MSE(y, tx, w)
    regularizer = lambda_*(np.linalg.norm(w)**2)
    return loss, loss + regularizer

# compute sigmoid function
def sigmoid(t):
    return np.exp(t)/(1 + np.exp(t))

# compute negative log likelihood loss of a model
def neg_log_loss(y, tx, w):
    prob = sigmoid(tx @ w)
    epsilon = 1e-15
    return -np.mean(y * np.log(np.clip(prob, epsilon, 1)) + (1 - y) * np.log(np.clip(1 - prob, epsilon, 1)))

# compute gradient of negative log likelihood loss function
def neg_log_gradient(y, tx, w):
    gradient = 1/y.shape[0] * (tx.T @ (sigmoid(tx @ w) - y))
    return gradient

# compute gradient of regularized negative log likelihood loss function
def neg_log_gradient_reg(y, lambda_, tx, w):
    gradient = neg_log_gradient(y, tx, w)
    return gradient + 2*lambda_*w

# train model using least squares 
def least_squares(y, tx):
    N = y.shape[0]
    y = np.reshape(y, (N,))
    w = np.linalg.solve(tx.T @ tx, tx.T @ y)
    MSE = 1/(2*N) * np.sum((y - (tx @ w))**2, 0)
    return w, MSE

# train model using gradient descent on the MSE loss function
def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    N = y.shape[0]
    y = np.reshape(y, (N,))
    threshold = 1e-8 # define convergence when difference between losses of two consecutive iters falls below this
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters): # iterating on guess for model weights
        gradient = compute_gradient_MSE(y, tx, w) # compute gradient over all data points
        loss = MSE(y, tx, w)
        w -= gamma*gradient # update w based on gradient computation
        ws.append(w)
        losses.append(loss)

        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break # convergence achieved, end loop and return weights/loss

    return w, losses[-1]

# train model using gradient descent on the MSE loss function
def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    N = y.shape[0]
    y = np.reshape(y, (N,))
    threshold = 1e-8 # define convergence when difference between losses of two consecutive iters falls below this
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters): # iterating on guess for model weights
        for minibatch_y, minibatch_tx in batch_iter(y, tx, 1): # iterating for batch subset of data points
            gradient = compute_gradient_MSE(minibatch_y, minibatch_tx, w) # compute gradient from subset
            loss = MSE(minibatch_y, minibatch_tx, w)
            w -= gamma*gradient # update weights with gradient
            ws.append(w)
            losses.append(loss)

            if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
                break # convergence achieved, end loop and return weights/loss

    return w, losses[-1]

# train model using ridge regression
def ridge_regression(y, tx, lambda_):
    N = y.shape[0]
    y = np.reshape(y, (N,))
    D = np.shape(tx)[1]
    lambda_prime = 2*N*lambda_  
    w = np.linalg.solve(((tx.T @ tx) + lambda_prime * np.identity(D)), (tx.T @ y))
    MSE, MSE_reg = MSE_regularized(y, tx, w, lambda_)
    return w, MSE

# train model using logistic regression
def logistic_regression(y, tx, initial_w, max_iters, gamma):
    N = y.shape[0]
    y = np.reshape(y, (N,))
    threshold = 1e-8 # define convergence when difference between losses of two consecutive iters falls below this
    w = initial_w
    losses = []
    
    for n_iter in range(max_iters): # iterating on guess for model weights

        gradient = neg_log_gradient(y, tx, w) # compute gradient of negative log likelihood loss at model weights
        loss = neg_log_loss(y, tx, w)
        w -= gamma*gradient # updating the weights based on gradient
        losses.append(loss)
        
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break # convergence achieved, end loop and return weights/loss

    return w, losses[-1]

# train model using regularized logistic regression
def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    N = y.shape[0]
    y = np.reshape(y, (N,))
    threshold = 1e-8 # define convergence when difference between losses of two consecutive iters falls below this
    w = initial_w
    losses = []
    
    for n_iter in range(max_iters): # iterating on guess for model weights

        gradient = neg_log_gradient_reg(y, lambda_, tx, w) # compute gradient of regularized negative log likelihood
        loss = neg_log_loss(y, tx, w)
        w -= gamma*gradient # update model weights based on gradient
        losses.append(loss)
        
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break # convergence achieved, end loop and return weights/loss

    return w, losses[-1]
