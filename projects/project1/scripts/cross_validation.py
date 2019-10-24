from plots import cross_validation_visualization
from implementations import *
from build_polynomial import *
import numpy as np
import matplotlib.pyplot as plt
 

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)

def cross_validation(y, tX, k_indices, k, lambda_, degree):
    """return the loss of ridge regression."""
    tr_indice = k_indices[~(np.arange(k_indices.shape[0]) == k)]
    tr_indice = tr_indice.reshape(-1)
    
    tX_te=tX[k_indices[k]]
    tX_tr=tX[tr_indice]
    
    y_te=y[k_indices[k]]
    y_tr=y[tr_indice]
    
    tX_te_poly=build_poly(tX_te,degree)
    tX_tr_poly=build_poly(tX_tr,degree)
    
    [w,loss_tr]=reg_logistic_regression(y_tr, tX_tr_poly, lambda_, np.ones((tX_tr_poly.shape[1], 1))/100, 50000, 0.1)
    loss_te=calc_loss_log(sigmoid(np.dot(tX_te_poly, w)), y_te) + (0.5*lambda_)*np.dot(w.T, w)
    
    return loss_tr, loss_te, w

def cross_validation_best_weight(y, tX, k_fold, degree, seed, lower_lambda, upper_lambda, name_to_add_in_path):
    
    lambdas = np.logspace(lower_lambda, upper_lambda, 10)
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    
    # define lists to store the loss of training data and test data
    rmse_tr = []
    rmse_te = []
    weights=[]
    best_lambdas=[]
    
    for d in range(1,degree):
        print("DEGREE = " + str(d))
        rmse_tr_l=[]
        rmse_te_l=[]
        rmse_te_l_for_box_plot=[]
        w_l=[]
        
        for lambda_ in lambdas:
            print("Lambda = "+str(lambda_))
            rmse_tr_k=[]
            rmse_te_k=[]
            w_k=[]
            
            for k in range(k_fold):
                print("K-Fold = ", k)
                cv_res=cross_validation(y, tX, k_indices, k, lambda_, d)
                rmse_tr_k.append(cv_res[0])
                rmse_te_k.append(cv_res[1])
                w_l.append(cv_res[2])
                
            rmse_tr_l.append(np.mean(rmse_tr_k))
            rmse_te_l.append(np.mean(rmse_te_k))
            rmse_te_l_for_box_plot.append(rmse_te_k)
            w_l.append(np.mean(w_l,axis=0))
            
        #plt.boxplot(rmse_te_l_for_box_plot)
        #plt.figure()
        best_index_l=np.argmin(rmse_te_l)
        cross_validation_visualization(lambdas, rmse_tr_l, rmse_te_l, name_to_add_in_path+str(d))
        rmse_tr.append(rmse_tr_l[best_index_l])
        rmse_te.append(rmse_te_l[best_index_l])
        weights.append(w_l[best_index_l])
        best_lambdas.append(lambdas[best_index_l])
        
    best_index_d=np.argmin(rmse_te)
    print("Test best error = "  + str(rmse_te[best_index_d]) + "for lambda = " + str(best_lambdas[best_index_d]) + "and degree = "+ str(best_index_d+1))
    
    return weights[best_index_d],rmse_te[best_index_d], best_index_d+1
