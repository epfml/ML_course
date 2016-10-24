# -*- coding: utf-8 -*-
import numpy as np
from costs import calculate_nll

"""
	Ridge regression functions
"""

def ridge_regression(y, tx, lambda_):
	"""implement ridge regression."""
	N = tx.shape[0]
	M = tx.shape[1]

	#w = np.dot(np.dot(np.linalg.inv(np.dot(tx.T, tx) + 2*N*lambda_*np.identity(M)), tx.T), y)
	
	A = np.dot(tx.T, tx) + 2*N*lambda_*np.identity(M)
	b = np.dot(tx.T, y)

	w = np.linalg.solve(A, b)

	return w

"""
	Utilities for logistic regression
"""

def sigmoid(t):
	"""apply sigmoid function on t."""
	return np.exp(t) / (1 + np.exp(t))

def calculate_gradient(y, tx, w):
	"""compute the gradient of loss."""
	return tx.T.dot(sigmoid(np.dot(tx, w)) - y)


"""
	Logistic regression using gradient descent method
"""

def learning_by_gradient_descent(y, tx, w, gamma):
	"""
	Do one step of gradient descent using logistic regression.
	Return the loss and the updated w.
	"""
	loss = calculate_loss(y, tx, w)

	grad = calculate_gradient(y, tx, w)

	w = w - alpha * grad
	return loss, w


def logistic_regression_gradient_descent(y, tx, gamma, max_iters):
	w = np.zeros((tx.shape[1], 1))	
	for iter in range(max_iters):
		# update w.
		w = learning_by_gradient_descent(y, tx, w, gamma)

	return w

""" 
	Logistic regression using Newton's method
"""

def calculate_hessian(y, tx, w):
	"""return the hessian of the loss function."""

	N = y.shape[0]
	xw = tx.dot(w)

	S = sigmoid(xw) * (1-sigmoid(xw)) * np.identity(N)

	return tx.T.dot(S.dot(tx))

def learning_by_newton_method(y, tx, w, gamma):
	"""
	Do one step on Newton's method.
	return the loss and updated w.
	"""
	grad = calculate_gradient(y, tx, w)
	H = calculate_hessian(y, tx, w)

	w = w - alpha * np.linalg.inv(H).dot(grad)
	return w

def logistic_regression_newton_method(y, tx, gamma, max_iters):
	w = np.zeros((tx.shape[1], 1))	
	for iter in range(max_iters):
		# update w.
		w = learning_by_newton_method(y, tx, w, gamma)

	return w

def logistic_regression(y, tx, gamma, max_iters):
	"""
	Do the logistic regression using gradient descent 
	or Newton's technique, return loss, w
	"""
#	w = logistic_regression_newton_method(y, tx, gamma, max_iters)
	w = logistic_regression_gradient_descent(y, tx, gamma, max_iters)

	loss = calculate_nll(y, tx, w)

	return loss, w

"""
	Penalized logistic regression
"""

def penalized_logistic_regression(y, tx, w, lambda_):
	"""return the gradient, and hessian."""
	loss = calculate_nll(y, tx, w) + lambda_ * w.T.dot(w)
	gradient = calculate_gradient(y, tx, w) + 2.0 * lambda_ * w
	H = calculate_hessian(y, tx, w) + 2.0 * lambda_
	return loss, gradient, H

def learning_by_penalized_gradient(y, tx, w, gamma, lambda_):
	"""
	Do one step of gradient descent, using the penalized logistic regression.
	Return the loss and updated w.
	"""
	loss, grad, H = penalized_logistic_regression(y, tx, w, lambda_)
	w = w - alpha * np.linalg.inv(H).dot(grad)

	return loss, w

def pen_logisitic_regression(y, tx, lambda_, gamma, max_iters):
	for iter in range(max_iter):
		loss, w = learning_by_penalized_gradient(y, tx, w, gamma, lambda_)

	return loss, w




