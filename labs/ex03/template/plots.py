# -*- coding: utf-8 -*-
"""a function of ploting figures."""
import numpy as np
from build_polynomial import *


def plot_fitted_curve(y, x, beta, degree, ax):
    """plot the fitted curve."""
    ax.scatter(x, y, color='b', s=12, facecolors='none', edgecolors='r')
    xvals = np.arange(min(x) - 0.1, max(x) + 0.1, 0.1)
    tx = np.c_[np.ones((len(xvals), 1)), build_poly(xvals, degree)]
    f = tx.dot(beta)
    ax.plot(xvals, f)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Polynomial degree " + str(degree))
