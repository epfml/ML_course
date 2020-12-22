"""Features engineering functions"""

import numpy as np


def classify(Data, classifier):
    """
    classifies the elements of Data in col_to_classify in 2 predefined classes 
    according to a threshold and returns the result.
    classifier is disctionary with each column to change and the corresponding classes
    """
    
    cols_to_classify = list(classifier.keys())
    for col in cols_to_classify:
        class1 = classifier[col][0]
        class2 = classifier[col][1]
        data_classified = Data[:,col].copy()
        threshold =  (class1 + class2)/2.0
        data_classified[data_classified < threshold] = class1
        data_classified[data_classified >= threshold] = class2
        Data[:,col] = data_classified.reshape(1,-1)
        
    return Data

def square_root(data, col_to_sqrt):
    """ 
    takes the square root of the elements of Data in col_to_sqrt.
    """
    data_sqrted = data[:,col_to_sqrt].copy()
    data_sqrted[data_sqrted >=  0] = np.sqrt(data_sqrted[data_sqrted >= 0])
    data[:,col_to_sqrt] = data_sqrted
    return data

def unif(data, col_to_unif):
    """
    divides the elements of X in col_to_divide by the absolute maximum 
    to bring the values between -1 and 1.
    """
    data_unif = data[:,col_to_unif].copy()
    absolute_max = np.amax(data_unif, axis = 0)
    data_unif = data_unif / absolute_max
    data[:,col_to_unif] = data_unif
    return data


def apply_log(data, columns):
    """
    applies the log to data and returns the new transformed data set
    """
    data_loged = data[:,columns].copy()
    data_loged[data_loged > 0] = np.log(data_loged[data_loged > 0])
    data_loged[data_loged == 0] = np.mean(data_loged[data_loged > 0])
    data[:,columns] = data_loged
    return data
