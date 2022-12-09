# -*- coding: utf-8 -*-
"""Some plot functions."""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from helper import build_distance_matrix


def plot_cluster(data, mu, colors, ax):
    """
    plot the cluster.

    Note that the dimension of the column vector `colors`
    should be the same as the number of clusters.
    """
    # build distance matrix.
    distance_matrix = build_distance_matrix(data, mu)
    # get the assignments for each point.
    assignments = np.argmin(distance_matrix, axis=1)
    for k_th in range(mu.shape[0]):
        rows = np.where(assignments == k_th)[0]
        data_of_kth_cluster = data[rows, :]
        ax.scatter(
            data_of_kth_cluster[:, 0],
            data_of_kth_cluster[:, 1],
            # works for clusters more than 3
            s=40,
            c=colors[k_th % len(colors)],
        )
    ax.grid()
    ax.set_xlabel("x")
    ax.set_ylabel("y")


def plot(data, mu, mu_old, out_dir):
    """plot."""
    colors = ["red", "blue", "black", "green", "yellow", "purple"]
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    plot_cluster(data, mu_old, colors, ax1)
    ax1.scatter(
        mu_old[:, 0], mu_old[:, 1], facecolors="y", edgecolors="y", s=120, marker="^"
    )

    ax2 = fig.add_subplot(1, 2, 2)
    plot_cluster(data, mu, colors, ax2)
    ax2.scatter(mu[:, 0], mu[:, 1], facecolors="y", edgecolors="y", s=120, marker="^")

    # matplotlib.rc('xtick', labelsize=5)
    # matplotlib.rc('ytick', labelsize=5)

    plt.tight_layout()
    plt.savefig(out_dir)
    plt.show()
    plt.close()


def plot_image_compression(original_image, image, assignments, mu, k):
    """plot histgram."""
    # init the plot
    fig = plt.figure()

    # visualization
    image_reconstruct = mu[assignments].reshape(original_image.shape)

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(original_image, cmap="Greys_r")
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(image_reconstruct, cmap="Greys_r")
    #     plt.draw()
    #     plt.pause(0.1)

    # ax3 = fig.add_subplot(2, 1, 2)

    # predifine colors
    # colors = np.array(
    #     [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1],
    #      [0.1, 0.1, 0.1], [1, 0.5, 0], [0, 0.5, 0], [0.5, 0.5, 0.5],
    #      [0.5, 0.25, 0], [0.5, 0, 0.5], [0, 0.5, 1], [1, 0, 0],
    #      [0, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [0.1, 0.1, 0.1],
    #      [1, 0.5, 0], [0, 0.5, 0], [0.5, 0.5, 0.5], [0.5, 0.25, 0],
    #      [0.5, 0, 0.5], [0, 0.5, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1],
    #      [1, 0, 1], [0, 1, 1], [0.1, 0.1, 0.1],
    #      [1, 0.5, 0], [0, 0.5, 0], [0.5, 0.5, 0.5], [0.5, 0.25, 0],
    #      [0.5, 0, 0.5], [0, 0.5, 1]])
    #
    # for k_th in range(k):
    #     rows, cols = np.where(assignments == k_th)
    #     hists, bins = np.histogram(image[rows], bins=10)
    #     width = 0.7 * (bins[1] - bins[0])
    #     center = (bins[:-1] + bins[1:]) / 2
    #     ax3.bar(center, hists, align='center', width=width,
    #             color=colors[k_th, :])
    #     ax3.plot(mu[k_th], 1, 'o', color=colors[k_th, :],
    #              linewidth=2, markersize=12, markerfacecolor=[1, 1, 1])
    # ax3.set_xlabel("x")
    # ax3.set_ylabel("y")
    # ax3.set_title("Histogram of clustered pixels.")
    plt.tight_layout()
    plt.savefig("image_compression.png")
    plt.show()
