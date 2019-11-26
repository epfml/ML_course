# -*- coding: utf-8 -*-
"""convert movielens100k.mat to the same data type of project 2."""
import numpy as np
from scipy.io import loadmat


def load_data():
    """load the mat data."""
    data = loadmat('movielens100k.mat')
    ratings = data['ratings']
    print("The data type of the 'ratings': {dt}".format(dt=type(ratings)))
    print("The shape of the 'ratings':{v}".format(v=ratings.shape))
    return ratings


def to_list(data):
    """save nz rating to list."""
    nz = np.nonzero(data)
    return ["r{}_c{},{}".format(nz_row + 1, nz_col + 1, data[nz_row, nz_col])
            for nz_row, nz_col in zip(*nz)]


def to_csv(data, path):
    """write data to csv file."""
    with open(path, "w") as f:
        f.write("\n".join(data))


if __name__ == '__main__':
    path = "movielens100k.csv"
    data = load_data()
    processed_data = to_list(data)
    to_csv(processed_data, path)
