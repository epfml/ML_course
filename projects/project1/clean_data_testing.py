import numpy as np
from poly import *

def clean_data_testing(x_te, labels, features, median_and_most_probable_class) :
    mask = [feature in features.keys() for feature in labels]
    x_te = x_te[:, mask]
    
    #for feature in features :
        #if feature not in ['SEQNO', '_PSU' , '_STSTR', '_STRWT', '_RAWRAKE', '_WT2RAKE', '_LLCPWT'] :
            #nan = median_and_most_probable_class[feature]
            #x_te[:, features[feature]] = np.nan_to_num(x_te[:,features[feature]], nan = median_and_most_probable_class[feature])
    x_te = np.nan_to_num(x_te, nan = -1)
    x_te = (x_te - np.mean(x_te, axis=0)) / np.std(x_te, axis=0)
    x_te = apply_pca(x_te)
    return x_te


def apply_pca(x_train):
    """
    Apply PCA to the data
    :param x_train: training data
    :return: pca applied data
    """
    mean = np.nanmean(x_train, axis=0)
    x_tilde = x_train - mean
    cov_matrix = np.cov(x_tilde, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov_matrix)
    eigvals = eigvals[::-1]
    eigvecs = eigvecs[:, ::-1]


    num_dimensions = 30
    W = eigvecs[:, 0 : num_dimensions]
    eg = eigvals[0 : num_dimensions]

    x_pca = np.dot(x_tilde, W)
    return x_pca