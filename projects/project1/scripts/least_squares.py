# -*- coding: utf-8 -*-
import numpy as np
from helpers import batch_iter
from costs import compute_loss

def compute_gradient(y, tx, w):
	"""Compute gradient for batch data."""
	N = y.shape[0]
	e = y - np.dot(tx, w)

	gradLw = -1/N * np.dot(tx.T, e)
	return gradLw


def least_squares_GD(y, tx, gamma, max_iters):
	"""Gradient descent algorithm."""

	w = np.zeros((1, tx.shape[1]))[0]

	for n_iter in range(max_iters):
		gradient = compute_gradient(y, tx, w)
		w = w - gamma * gradient

	loss = compute_loss(y, tx, w)
        
	return loss, w


def least_squares_SGD(y, tx, gamma, max_iters):
	"""Stochastic gradient descent algorithm."""
	batch_size = 50
	w = np.zeros((1, tx.shape[1]))[0]

	for n_iter in range(max_iters):
		y_, tx_ = batch_iter(y, tx, batch_size).__next__()
		gradient = compute_gradient(y_, tx_, w)
		w = w - gamma * gradient

	loss = compute_loss(y, tx, w)

	return loss, w


def least_squares(y, tx):
	"""calculate the least squares solution."""
	#w = np.dot(np.dot(np.linalg.inv(np.dot(tx.T, tx)), tx.T), y)

	A = np.dot(tx.T, tx)
	b = np.dot(tx.T, y)

	w = np.linalg.solve(A, b)
	loss = compute_loss(y, tx, w)

	return loss, w
