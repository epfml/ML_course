import numpy as np
import matplotlib.pyplot as plt
from helpers import *
from utils import remove_features, find_key_by_value
from poly import *
#Defining some constants
import numpy as np
import matplotlib.pyplot as plt
from helpers import *
from utils import remove_features, find_key_by_value, create_pca, upsample_class_1_to_percentage, build_poly
from normalization import z_score_normalization, min_max_normalization, normalize_data

from config import dictionary_features, category_features


#Defining some constants

median_and_most_probable_class = {}
ACCEPTABLE_NAN_PERCENTAGE = 0.1
ACCEPTABLE_NAN_ROW_PERCENTAGE = 0.4

def clean_train_data(x_train,y_train, labels, up_sampling_percentage, degree):
    """
    Cleaning data
    :param x_train: training data
    :return: cleaned data
    """
    y_train[y_train == -1] = 0
    x_train, y_train = upsample_class_1_to_percentage(x_train, y_train, up_sampling_percentage)
    
    #Removing the first label which is the id
    features_number = x_train.shape[1]
    features = list(dict.fromkeys(labels))  
    features = [features[i]  for i in range(features_number)]
    features = {word: index for index, word in enumerate(features)}
    
    #We handle the date and rescale some of the features
    #x_train = handling_data(x_train, features)

    #Removing columns with more than ACCEPTABLE_NAN_PERCENTAGE of NaN values
    mask_nan_columns = [(np.count_nonzero(np.isnan(x_train[:, i]))/x_train.shape[0]) <= ACCEPTABLE_NAN_PERCENTAGE for i in range (features_number)]
    x_train = x_train[:, mask_nan_columns]

    #Creating features list
    features = list(dict.fromkeys(labels))  
    features = [features[i]  for i in range (features_number) if mask_nan_columns[i]]
    features = {word: index for index, word in enumerate(features)}
   
    
    
    #We remove the features that are not useful
    
    x_train = handle_nan(x_train, features)
    
    #normalize the data
    x_train = normalize_data(x_train)
    x_train, W, mean = create_pca(x_train)
    poly_x = build_poly(x_train, degree)

    return poly_x, y_train,  features, median_and_most_probable_class, W, mean


def handle_nan(x_train, features) :
    
    # Replace NaN in categorical features with the median value
    for feature in features:
        
        median_value = np.nanmedian(x_train[:, features[feature]])
        median_and_most_probable_class[feature] = median_value
        x_train[: ,features[feature]] = np.nan_to_num(x_train[:,features[feature]], nan = median_value)
        
    return x_train

def drop_na_row(x, y) :
    x = x.copy()
    y = y.copy()
    mask_nan_rows = [(np.count_nonzero(np.isnan(x[i, :]))/x.shape[1]) <= ACCEPTABLE_NAN_ROW_PERCENTAGE for i in range(x.shape[0])]

    x = x[mask_nan_rows,:]
    y = y[mask_nan_rows]
    return x,y

        



