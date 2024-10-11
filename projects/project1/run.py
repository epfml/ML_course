import numpy as np
import matplotlib.pyplot as plt
from helpers import *
from cleaning_data import *

def run():

    DATA_PATH = 'projects\project1\data\dataset\dataset'
    x_train, x_test, y_train, train_ids, test_ids, labels =  load_csv_data(DATA_PATH, sub_sample=False)
    labels.pop(0)

    # Clean data
    x_train_clean = clean_data_x(x_train, labels)

    np.savetxt('projects\project1\data\dataset\dataset\\x_train_cleaned.csv', x_train_clean, delimiter=',')

run()