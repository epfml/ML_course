# -*- coding: utf-8 -*-
"""a function of ploting figures."""
import numpy as np
import matplotlib.pyplot as plt


def cross_validation_visualization(lambds, rmse_tr, rmse_te):
    """visualization the curves of rmse_tr and rmse_te."""
    plt.semilogx(lambds, rmse_tr, marker=".", color="b", label="train error")
    plt.semilogx(lambds, rmse_te, marker=".", color="r", label="test error")
    plt.xlabel("lambda")
    plt.ylabel("r mse")
    # plt.xlim(1e-4, 1)
    plt.title("cross validation")
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig("cross_validation")


def bias_variance_decomposition_visualization(degrees, rmse_tr, rmse_te):
    """visualize the bias variance decomposition."""
    rmse_tr_mean = np.expand_dims(np.mean(rmse_tr, axis=0), axis=0)
    rmse_te_mean = np.expand_dims(np.mean(rmse_te, axis=0), axis=0)
    plt.plot(degrees, rmse_tr.T, linestyle="-", color=([0.7, 0.7, 1]), linewidth=0.3)
    plt.plot(degrees, rmse_te.T, linestyle="-", color=[1, 0.7, 0.7], linewidth=0.3)
    plt.plot(degrees, rmse_tr_mean.T, "b", linestyle="-", label="train", linewidth=3)
    plt.plot(degrees, rmse_te_mean.T, "r", linestyle="-", label="test", linewidth=3)
    plt.xlim(1, 9)
    plt.ylim(0.2, 0.7)
    plt.xlabel("degree")
    plt.ylabel("error")
    plt.legend(loc=1)
    plt.title("Bias-Variance Decomposition")
    plt.savefig("bias_variance")
