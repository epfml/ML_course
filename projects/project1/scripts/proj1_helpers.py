# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import csv
import numpy as np


def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y=='b')] = -1
    
    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids

def load_data_jet_number(data_path,number,sub_sample=False):
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    
    tx_num = x[:,2:]
    tx_num = tx_num[tx_num[:,22]==number]
    
    #remove useless variables 
    if number==0:
        tx_num=np.delete(tx_num,[4,5,6,12,22,23,24,25,26,27,28,29],1) #Without col 29 loss is slightly better as this col is always 0. 
    elif number==1:
        tx_num=np.delete(tx_num,[4,5,6,12,22,26,27,28],1)
    elif number==2:
        tx_num=np.delete(tx_num,22,1)
    elif number==3:
        tx_num=np.delete(tx_num,22,1)
    y = y[x[:,24]==number]
    y_num = np.ones(len(y))
    y_num[np.where(y=='b')] = -1
    

    ids = x[x[:,24]==number]
    ids = ids[:, 0].astype(np.int)

    if sub_sample:
        y_num = y_num[::50]
        tx_num = tx_num[::50]
        ids = ids[::50]

    return y_num, tx_num, ids


def predict_labels(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.dot(data, weights)
    print(y_pred[0:100])
    y_pred[np.where(y_pred <= 0.5)] = -1
    y_pred[np.where(y_pred > 0.5)] = 1
    
    return y_pred


def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})
