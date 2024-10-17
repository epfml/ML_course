import numpy as np
from poly import *
from utils import apply_pca_given_components

def clean_test_data(x_te, labels, features, median_and_most_probable_class, mean, W) :
    
    mask = [feature in features.keys() for feature in labels]
    x_te = x_te[:, mask]
    
    #for feature in features :
        #if feature not in ['SEQNO', '_PSU' , '_STSTR', '_STRWT', '_RAWRAKE', '_WT2RAKE', '_LLCPWT'] :
            #nan = median_and_most_probable_class[feature]
            #x_te[:, features[feature]] = np.nan_to_num(x_te[:,features[feature]], nan = median_and_most_probable_class[feature])

    x_te = np.nan_to_num(x_te, nan = -1)
    x_te = (x_te - np.mean(x_te, axis=0)) / np.std(x_te, axis=0)
    x_te = apply_pca_given_components(x_te, mean, W)
    return x_te

