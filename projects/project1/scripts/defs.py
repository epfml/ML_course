##############################################################
#
#DEFINITION OF ALL THE FUNCTIONS WE WILL NEED LATER ON
#
##############################################################
import numpy as np
from implementations import *
from costs import *

def compute_gradient(y, tx, w):
    """Compute the gradient."""
    e = y - tx.dot(w)
    grad = -tx.T.dot(e) / len(e)
    return grad, e

def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(err)
    return grad, err

def calculate_gradient(y, tx, w):
    """compute the gradient of loss."""
    pred = sigmoid(tx.dot(w))
    grad = tx.T.dot(pred - y)
    return grad

def load_data_jet_number(data_path,number,sub_sample=False):
    '''Function to load data depending on the number "jet" '''
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    
    tx_num = x[:,2:]
    tx_num = tx_num[tx_num[:,22]==number]
    
    #remove useless variables 
    if number==0:
        tx_num=np.delete(tx_num,[4,5,6,12,22,23,24,25,26,27,28,29],1) #Without col 29 loss is slightly better as this col is always 0. 
    elif number==1:
        tx_num=np.delete(tx_num,[4,5,6,12,22,26,27,28],1)
    elif number==2:
        tx_num=np.delete(tx_num,22,1)
    elif number==3:
        tx_num=np.delete(tx_num,22,1)
    y = y[x[:,24]==number]
    y_num = np.ones(len(y))
    y_num[np.where(y=='b')] = -1
    

    ids = x[x[:,24]==number]
    ids = ids[:, 0].astype(np.int)

    if sub_sample:
        y_num = y_num[::50]
        tx_num = tx_num[::50]
        ids = ids[::50]

    return y_num, tx_num, ids

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    poly = x
    for deg in range(2, degree+1):
        poly = np.c_[poly, np.power(x, deg)]
    one=np.ones([poly.shape[0],1])
    return np.c_[one,poly]

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    n_row = y.shape[0]
    interval = int(n_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(n_row)

    k_indices = [indices[i * interval : (i+1) * interval] for i in range(k_fold)]
    return np.array(k_indices)

def cross_validation(y, tX, k_indices, k, lambda_, degree):
    length = k_indices.shape[0]
    train_indice = k_indices[~(np.arange(length) == k)]
    train_indice = train_indice.reshape(-1)
    
    tX_test=tX[k_indices[k]]
    tX_train=tX[train_indice]
    
    y_test=y[k_indices[k]]
    y_train=y[train_indice]
    
    tX_test_poly=build_poly(tX_test,degree)
    tX_train_poly=build_poly(tX_train,degree)
    
    w,loss_tr=ridge_regression(y_train,tX_train_poly,lambda_)
    loss_te=compute_loss(y_test,tX_test_poly,w,type="mae")
    
    return loss_tr, loss_te, w

def cross_validation_best_weight(y, tX, k_fold, degree, seed, lower_lambda, upper_lambda, name_to_add_in_path):
    number_lambdas = 30
    lambdas = np.logspace(lower_lambda, upper_lambda, number_lambdas)
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    
    # define lists to store the loss of training data and test data
    mae_tr = []
    mae_te = []
    weights=[]
    best_lambdas=[]
    
    for d in range(1,degree):
        print("DEGREE = " + str(d))
        mae_tr_l=[]
        mae_te_l=[]
        mae_te_l_for_box_plot=[]
        w_l=[]
        
        for lambda_ in lambdas:
            print("Lambda = "+str(lambda_))
            mae_tr_k=[]
            mae_te_k=[]
            w_k=[]
            
            for k in range(k_fold):
                print("K-Fold = ", k)
                cv_res=cross_validation(y, tX, k_indices, k, lambda_, d)
                mae_tr_k.append(cv_res[0])
                mae_te_k.append(cv_res[1])
                w_l.append(cv_res[2])
                
            mae_tr_l.append(np.mean(mae_tr_k))
            mae_te_l.append(np.mean(mae_te_k))
            mae_te_l_for_box_plot.append(mae_te_k)
            w_l.append(np.mean(w_l,axis=0))
            
        best_index_l=np.argmin(mae_te_l)
        mae_tr.append(mae_tr_l[best_index_l])
        mae_te.append(mae_te_l[best_index_l])
        weights.append(w_l[best_index_l])
        best_lambdas.append(lambdas[best_index_l])
        
    best_index_d=np.argmin(mae_te)
    print("Test best error = "  + str(mae_te[best_index_d]) + " for lambda = " + str(best_lambdas[best_index_d]) + " and degree = "+ str(best_index_d+1))
    
    return weights[best_index_d],mae_te[best_index_d], best_index_d+1