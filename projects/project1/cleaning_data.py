import numpy as np
import matplotlib.pyplot as plt
from helpers import *
from utils import remove_features, find_key_by_value
from poly import *
#Defining some constants
import numpy as np
import matplotlib.pyplot as plt
from helpers import *
from utils import remove_features, find_key_by_value
from normalization import z_score_normalization, min_max_normalization

from config import dictionary_features, category_features


#Defining some constants
median_and_most_probable_class = {}

ACCEPTABLE_NAN_PERCENTAGE = 0


def clean_data_x(x_train, labels):
    """
    Cleaning data
    :param x_train: training data
    :return: cleaned data
    """
    #Removing the first label which is the id
    features_number = x_train.shape[1]
    features = list(dict.fromkeys(labels))  
    features = [features[i]  for i in range(features_number)]
    features = {word: index for index, word in enumerate(features)}
    
    #We handle the date and rescale some of the features
    #x_train = handling_data(x_train, features)
    
    #Removing columns with more than ACCEPTABLE_NAN_PERCENTAGE of NaN values
    mask_nan_columns = [(np.count_nonzero(np.isnan(x_train[:, i]))/x_train.shape[0]) <= ACCEPTABLE_NAN_PERCENTAGE for i in range (0, features_number)]
    x_train = x_train[:, mask_nan_columns]

    #Creating features list
    features = list(dict.fromkeys(labels))  
    features = [features[i]  for i in range (features_number) if mask_nan_columns[i]]
    features = {word: index for index, word in enumerate(features)}
   
    
    
    #We remove the features that are not useful
    #features, x_train = remove_features(x_train, ['WEIGHT2', 'HEIGHT3', 'SEQNO', '_PSU' , '_STSTR', '_STRWT', '_RAWRAKE', '_WT2RAKE', 'DISPCODE','_LLCPWT','IYEAR','IMONTH','INTERNET','FMONTH','IDATE'  ], features)
   
    x_train = handle_nan(x_train, features)
    x_train, features = handle_correlation(x_train, features)
    #normalize the data
    x_train = normalize_data(x_train, features)
    x_train = apply_pca(x_train)

    return x_train, features, median_and_most_probable_class

def handling_data(x_train, features):
    """
    Handling and modifying data because of special values and scaling some values
    :param x_train: training data
    :return: modified data
    """
    #Normalizing data
    for feature in features.keys() :
    
        dict_special_value_handle = dictionary_features[feature]


        for special_value in dict_special_value_handle.keys() :
            #For special values meaning not sure, we change by the median
            if special_value == 'dont_know_not_sure' :
                x_train[x_train[:, features[feature]] == dict_special_value_handle[special_value], features[feature]] = np.nanmedian(x_train[:, features[feature]])
                
            #For special values meaning none, we change by 0
            elif special_value == 'never' or special_value == 'none'  :
                x_train[x_train[:, features[feature]] == dict_special_value_handle[special_value], features[feature]] = 0
                    
            #For special values meaning refused, we change by NaN
            elif special_value == 'refused' :
                x_train[x_train[:, features[feature]] == dict_special_value_handle[special_value], features[feature]] = np.nan
    
    x_train = day_week_month_rescale(x_train, 'STRENGTH', 1, features)
    x_train = day_week_month_rescale(x_train, 'ALCDAY5', 1, features)
    x_train = day_week_month_rescale(x_train, 'FRUIT1', 2, features)
    x_train = day_week_month_rescale(x_train, 'FVBEANS', 2, features)
    x_train = day_week_month_rescale(x_train, 'FVORANG', 2, features)
    x_train = day_week_month_rescale(x_train, 'FVGREEN', 2, features)
    x_train = day_week_month_rescale(x_train, 'VEGETAB1', 2, features)
    x_train = day_week_month_rescale(x_train, 'FRUITJU1', 2, features)

    return x_train

def day_week_month_rescale(x, feature_name, scaling_mode, features):
    """
    Rescale the values of the feature_name
    :param x: training data
    :param feature_name: feature name
    :param scaling_mode: scaling mode
    :param features: features list
    :return: modified data
    """
    if feature_name in features.keys() :
    
        mask_three_hundred = x[:, features[feature_name]] > 300
        mask_two_hundred = (x[:, features[feature_name]] >= 200) & (x[:, features[feature_name]] < 300)
        mask_one_hundred = (x[:, features[feature_name]] >= 100) &(x[:, features[feature_name]] < 200) 
            
        if scaling_mode == 1 : 
            x[mask_one_hundred , features[feature_name]] = (x[mask_one_hundred, features[feature_name]] -100)*4.33
            x[mask_two_hundred, features[feature_name]] = (x[mask_two_hundred, features[feature_name]] -200)
            
        elif scaling_mode == 2:
            x[x[:, features[feature_name]] == 300, features[feature_name]] = 0
            x[mask_one_hundred , features[feature_name]] = (x[mask_one_hundred, features[feature_name]] -100)*(4.33*7)
            x[mask_two_hundred, features[feature_name]] = (x[mask_two_hundred, features[feature_name]] -200)*(4.33)
            x[mask_three_hundred, features[feature_name]] = (x[mask_three_hundred, features[feature_name]] -300)

    return x
    
def handle_nan2(x_train, features) :
    # Replace NaN in categorical features with the median value
    x_train = np.nan_to_num(x_train, nan = -1)
    return x_train


def handle_nan(x_train, features) :
    # Replace NaN in categorical features with the median value
    for feature in features:
        median_value = np.nanmedian(x_train[:, features[feature]])
        median_and_most_probable_class[feature] = median_value
        x_train[: ,features[feature]] = np.nan_to_num(x_train[:,features[feature]], nan = median_value)

  
    return x_train
    
def handle_correlation(x_train, features):
    """
    Handling correlation between features
    :param x_train: training data
    :param features: features list
    :return: modified and correlation handled data
    """
    # Compute the correlation matrix
    features_correlation = np.corrcoef(x_train, rowvar=False)

    # Find the features that are highly correlated
    correlation_limit = 0.5
    correlation_tuple_list = []
    correlation_list = []

    for i in range(x_train.shape[1]) : 
        for j in range(i, x_train.shape[1]) : 
            if i != j and features_correlation[i,j] >= correlation_limit : 
                correlation_tuple_list.append((find_key_by_value(features, i) , find_key_by_value(features, j)))
                correlation_list.append(find_key_by_value(features, i))
                correlation_list.append(find_key_by_value(features, j))
    
    # Use np.unique to get counts of each element
    correlation_list = np.array(correlation_list)
    unique_elements, counts = np.unique(correlation_list, return_counts=True)
    count_elements = dict(zip(unique_elements, counts))

    features_to_remove = set()
    # Iterate through the correlation tuples
    for feature1, feature2 in correlation_tuple_list:
        # Compare the counts of the two features
        if count_elements[feature1] > count_elements[feature2]:
            features_to_remove.add(feature2)
        else:
            features_to_remove.add(feature1)

    # Remove the features from the features dictionary and x_train_modified
    features, x_train = remove_features(x_train, list(features_to_remove), features)

    # Update the features dictionary to reflect the new indices
    features = {feature: i for i, feature in enumerate(features.keys())}

    return x_train, features

def normalize_data(x, features) :
    x = (x - np.mean(x, axis=0)) / np.std(x, axis=0)
    return x

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


    num_dimensions = 20
    W = eigvecs[:, 0 : num_dimensions]
    eg = eigvals[0 : num_dimensions]

    x_pca = np.dot(x_tilde, W)
    return x_pca