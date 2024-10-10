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