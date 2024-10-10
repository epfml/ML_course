import numpy as np


def min_max_normalization(x):
    """
    Min-max normalization
    :param x: variable x
    :return: list of numbers
    """
    min_val = min(x)
    max_val = max(x)
    return (x - min_val) / (max_val - min_val)

def z_score_normalization(x):
    """
    Z-score normalization
    :param x: variable x 
    :return: list of numbers
    """
    mean_val = np.mean(x)
    std_val = np.std(x)
    return (x - mean_val) / std_val