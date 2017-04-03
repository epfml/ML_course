# -*- coding: utf-8 -*-
import numpy as np
from costs import calculate_nll, compute_loss
from least_squares import least_squares

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
	loss = compute_loss(y, tx, w)

	return loss, w

"""
	Utilities for logistic regression
"""

def sigmoid(t):
	"""apply sigmoid function on t."""
	precLim = 50

	pre = t 
	for i in range (t.shape[0]):
		elem = pre[i]
		if elem>precLim:
			pre[i] = 1
		else:
			if elem<-precLim:
				pre[i] = 0
			else:
				pre[i] = np.exp(elem) / (1 + np.exp(elem))

	return pre.reshape((pre.shape[0]))

def calculate_gradient(y, tx, w):
	"""compute the gradient of loss."""

	sig = sigmoid(np.dot(tx, w))
	sigy = sig - y

	ret = tx.T.dot(sigy)
	return ret

def calculate_hessian(y, tx, w):
	"""return the hessian of the loss function."""	
	N = y.shape[0]
	xw = tx.dot(w)

	S = (sigmoid(xw) * (1-sigmoid(xw))).reshape((N, 1)) * tx

	return np.dot(tx.T, S)


"""
	Logistic regression learning phase
"""

def learning_by_gradient_descent(y, tx, w, gamma):
	"""
	Do one step of gradient descent using logistic regression.
	Return the loss and the updated w.
	"""
	loss = calculate_loss(y, tx, w)
	grad = calculate_gradient(y, tx, w)

	w = w - gamma * grad
	return loss, w

def logistic_regression_gradient_descent(y, tx, gamma, max_iters):
	w = np.zeros((tx.shape[1], 1))	
	loss = 0
	for iter in range(max_iters):
		loss, w = learning_by_gradient_descent(y, tx, w, gamma)

	return loss, w


def default_logistic_regression(y, tx, w):
	"""return the loss, gradient, and hessian."""
	loss = calculate_nll(y, tx, w)
	gradient = calculate_gradient(y, tx, w)
	H = calculate_hessian(y, tx, w)
	return loss, gradient, H

def learning_by_newton_method(y, tx, w, gamma):
	"""
	Do one step on Newton's method.
	return the loss and updated w.
	"""

	loss, grad, H = default_logistic_regression(y, tx, w)
	
	hgrad = np.linalg.inv(H).dot(grad)
	w = w - gamma * hgrad
	return loss, w

def logistic_regression_newton_method(y, tx, gamma, max_iters):
	w = np.zeros((tx.shape[1]))
	loss = 0
	for iter in range(max_iters):
		loss, w = learning_by_newton_method(y, tx, w, gamma)
	
	return loss, w

""" 
	Logistic regression
"""

def logistic_regression(y, tx, gamma, max_iters):
	"""
	Do the logistic regression using gradient descent 
	or Newton's technique, return loss, w
	"""
	loss, w = logistic_regression_newton_method(y, tx, gamma, max_iters)
#	loss, w = logistic_regression_gradient_descent(y, tx, gamma, max_iters)

	return loss, w

"""
	Penalized logistic regression learning
"""

def penalized_logistic_regression(y, tx, w, lambda_):
	"""return the loss, gradient, and hessian."""
	loss = (calculate_nll(y, tx, w) + lambda_ * w.T.dot(w))
	gradient = calculate_gradient(y, tx, w) + 2.0 * lambda_ * w
	H = calculate_hessian(y, tx, w) + 2.0 * lambda_
	
	return loss, gradient, H

def learning_by_penalized_gradient(y, tx, w, gamma, lambda_):
	"""
	Do one step of gradient descent, using the penalized logistic regression.
	Return the loss and updated w.
	"""
	loss, grad, H = penalized_logistic_regression(y, tx, w, lambda_)

	hgrad = np.linalg.inv(H).dot(grad)

	w = w - gamma * hgrad

	return loss, w

"""
	Penalized logistic regression
"""

def pen_logisitic_regression(y, tx, lambda_, gamma, max_iters):
	w = np.zeros((tx.shape[1]))
	loss = 0
	for iter in range(max_iters):
		loss, w = learning_by_penalized_gradient(y, tx, w, gamma, lambda_)

	return loss, w




