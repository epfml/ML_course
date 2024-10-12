import numpy as np

def calculate_excess_kurtosis(arr):
    mean = np.nanmean(arr)
    std = np.nanstd(arr)
    fourth_moment = np.nanmean((arr - mean) ** 4)
    kurtosis = ((fourth_moment) / (std ** 4)) - 3 
    return kurtosis

def calculate_skewness(arr):
    mean = np.nanmean(arr)
    std = np.nanstd(arr)
    third_moment = np.nanmean((arr - mean) ** 3)
    skewness = ((third_moment) / (std ** 3))
    return skewness

def box_cox(arr, _lambda) :
    if np.any(arr) >= 0:
        return -1
    if _lambda == 0:
        return np.log(arr)
    else :
        return (np.power(arr, _lambda) - 1) / _lambda
        
def IQR(arr) :
    Q1 = np.nanpercentile(arr, 25)
    Q3 = np.nanpercentile(arr, 75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return lower_bound, upper_bound

def f1_score(y_true, y_pred):
    """
    Compute the F1 score
    :param y_true: true labels
    :param y_pred: predicted labels
    Maybe we should use only one column in the y_true and y_pred
    :return: F1 score
    """
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == -1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == -1))
    return tp / (tp + 0.5 * (fp + fn))

def accuracy(y_true, y_pred):
    """"
    Compute the accuracy
    :param y_true: true labels
    :param y_pred: predicted labels
    Maybe we should use only one column in the y_true and y_pred
    :return: accuracy
    """
    return np.sum(y_true == y_pred) / len(y_true)