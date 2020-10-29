# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import csv
import numpy as np
import implementations


def standardize_data(X):
    """Standardize the cleaned data set X."""
    means = np.mean(X, axis=0)
    data_sm = X - means
    std = np.std(data_sm, axis=0)
    standard_data = data_sm / std
    return standard_data, std, means

def clean_data(input_data, mean=False):
    """ Replaces -999 by most frequent value of column or mean if mean=True """
    current_col = input_data[:, 0]

    if -999.0 in current_col:
        indices_to_change = (current_col == -999.0)
        if mean:
            curr_mean = np.mean(current_col[~indices_to_change])
            current_col[indices_to_change] = curr_mean
        else:
            (values,counts) = np.unique(current_col[~indices_to_change], return_counts=True)
            ind=np.argmax(counts)
            current_col[indices_to_change] = values[ind] if len(values) > 0 else 0

    return input_data


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    poly = np.ones((len(x), 1))
    for deg in range(1, degree + 1):
        poly = np.c_[poly, np.power(x, deg)]
    return poly

def seperate_PRI_jet_num(X):
    ''' 
    Finds the indices of rows that have jet 0,1,2,3 and returns it in an array of 4 arrays.
    '''
    
    rows = [[], [], [], []]
    for ind, item in enumerate(X):
        rows[int(item[22])].append(ind)
    return rows

def split_data(x, y, ratio, seed=1):
    """
    split the dataset based on the split ratio. If ratio is 0.8 
    you will have 80% of your data set dedicated to training 
    and the rest dedicated to testing
    """
    # set seed
    np.random.seed(seed)
    
    indices = np.random.permutation(np.arange(len(y)))
    
    splits = [int(len(y)*ratio)]    #.cum() if severals
    x_train, x_test = np.split(x[indices], splits)
    y_train, y_test = np.split(y[indices], splits)
    
    return x_train, y_train, x_test, y_test

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

#********************************************************

"""Function used to compute the loss."""

def compute_mse(y, tx, w):
    """Calculate the mse for vector e."""
    e = y - tx.dot(w)
    return 0.5*np.mean(e**2)


def compute_mae(y, tx, w):
    e = y - tx.dot(w)
    return np.mean(np.abs(e))

def compute_rmse(y, tx, w):
    mse = compute_mse(y, tx, w)
    return np.sqrt(2*mse)

def compute_loss(y, tx, w):
    """Calculate the loss.
    You can calculate the loss using mse or mae.
    """
    #MSE
    e = y - tx.dot(w)
    N = len(y)
    loss = (0.5/N) * e.T.dot(e)
    return loss
#********************************************************

def accuracy(y_guessed, y_te):
    """This method returns the percentage of correctness after prediction"""
    R = 0
    for i in range(len(y_guessed)):
        if (y_guessed[i] == y_te[i]):
            R = R + 1
    return 100 * R / len(y_guessed)



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
#********************************************************

"""Gradient Descent"""

def compute_gradient(y, tx, w):
    """Compute the gradient."""
    e = y - tx.dot(w)
    gradient = (- 1/len(y)) * tx.T.dot(e)
    return gradient, compute_mse(y, tx, w)

def gradient_descent(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        g, loss = compute_gradient(y, tx, w)
        w = w - gamma * g
        # store w and loss
        ws.append(w)
        losses.append(loss)
        if n_iter % 500 == 0:
            print("Current iteration={i}, w={w} loss={l}".format(i=n_iter, w=w, l=loss))
    return ws, losses
#********************************************************            

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
                loss = compute_mse(y, tx, w)
                w = w - gamma * g
                
                #append ws and losses
                ws.append(w)
                losses.append(loss)
                
                if n_iter % 500 == 0:
                    print("Current iteration={i}, w={w} loss={l}".format(i=n_iter, w=w, l=loss))
                
    return ws, losses
#********************************************************
"""Helpers for logistic regression"""

def sigmoid(t):
    """apply the sigmoid function on t."""
    return 1 / (1 + np.exp(-t))
    
def calculate_loss_logistic(y, tx, w):
    """compute the loss: negative log likelihood."""
    
    pred = sigmoid(tx.dot(w))
    loss = - ( y.T.dot(np.log(pred)) + (1 - y).T.dot(np.log(1 - pred)) )
    loss = np.squeeze(loss)
    return loss

def calculate_gradient_logistic(y, tx, w):
    """compute the gradient of loss."""
    pred = sigmoid(tx.dot(w))
    g = tx.T.dot(pred - y)
    return g

def learning_by_gradient_descent(y, tx, w, gamma):
    """
    Do one step of gradient descent using logistic regression.
    Return the loss and the updated w.
    """
    # compute the loss:
    loss = calculate_loss_logistic(y, tx, w)
    # compute the gradient:
    g = calculate_gradient_logistic(y, tx, w)
    # update w:
    w = w - gamma * g
    return w, loss



def penalized_logistic_regression(y, tx, w, lambda_):
    """return the loss, gradient"""
    # return loss, gradient, and Hessian:
    loss, gradient, hessian = logistic_regression(y, tx, w)
    
    loss = loss + lambda_ * w.T.dot(w).squeeze()
    gradient = gradient + 2 * lambda_ * w
    return loss, gradient

def learning_by_penalized_gradient(y, tx, w, gamma, lambda_):
    """
    Do one step of gradient descent, using the penalized logistic regression.
    Return the loss and updated w.
    """
    # return loss, gradient:
    loss, gradient = penalized_logistic_regression(y, tx, w, lambda_)
    # update w:
    w = w - gamma * gradient
    return w, loss



def predict_labels(weights, data):
    """Returns class predictions given weights and data matrix"""
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1
    return y_pred




def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y=='b')] = -1
    
    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids



def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})
