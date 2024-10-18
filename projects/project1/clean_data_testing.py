import numpy as np
from poly import *
from utils import apply_pca_given_components, build_poly
from normalization import normalize_data


def clean_test_data(x_te, labels, features, median_and_most_probable_class, mean, W, degree) :
    
    mask = [feature in features.keys() for feature in labels]
    x_te = x_te[:, mask]
    
    for feature in features :
        nan = median_and_most_probable_class[feature]
        x_te[:, features[feature]] = np.nan_to_num(x_te[:,features[feature]], nan = median_and_most_probable_class[feature])

    x_te = normalize_data(x_te)
    x_te = apply_pca_given_components(x_te, mean, W)
    poly_x_te = build_poly(x_te, degree)
    return poly_x_te

