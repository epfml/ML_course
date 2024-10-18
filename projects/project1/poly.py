import numpy as np


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree.

    Args:
        x: numpy array of shape (N,D), N is the number of samples and D the number of features
        degree: integer.

    Returns:
        poly: numpy array of shape (N,d+1)

    >>> build_poly(np.array([0.0, 1.5]), 2)
    array([[1.  , 0.  , 0.  ],
           [1.  , 1.5 , 2.25]])
    """
    N, D = x.shape
    poly = np.ones((N, (degree + 1) * D))  # Initialize polynomial feature matrix

    # Expand each feature in X to its powers from 0 to the given degree
    for i in range(D):
        for j in range(degree + 1):
            poly[:, i * (degree + 1) + j] = np.power(x[:, i], j)



    return poly