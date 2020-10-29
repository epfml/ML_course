import numpy as np
import os.path
from implementations import *
from proj1_helpers import *
from Feature_expansion import *

# Check if data exists and import it

# Using os.path.join to avoid being OS dependent
train_data_folder_path = os.path.join("data", "train.csv")
test_data_folder_path = os.path.join("data", "test.csv")

if os.path.exists(train_data_folder_path) and os.path.exists(test_data_folder_path):
    print('loading data from /data directory')
    y, tX, ids_tr = load_csv_data('data/train.csv')

elif os.path.exists('train.csv') and os.path.exists('test.csv'):
    print('loading data from main directory')
    y, tX, ids_tr = load_csv_data('train.csv')

else:
    print('There is no train/test file in the directory try importing it and running again')
    sys.exit()

lambdas = [6.16e-05, 2.42e-05, 3.3e-05, 1e-05]
degrees = [5, 5, 5, 4]

classifier_0 = {11 : [-1,1]}
classifier_1 = {11 : [-1,1]}
classifier_2 = {11 : [-1, 1], 12: [0, 1]}
classifier_3 = {11 : [-1, 1], 12: [0, 1]}

classifier = [classifier_0, classifier_1, classifier_2, classifier_3]

unif_0 = [14, 15, 17, 18, 20]
unif_1 = [14, 15, 17, 18, 20, 25]
unif_2 = [14, 15, 17, 18, 20, 25, 28]
unif_3 = [14, 15, 17, 18, 20, 25, 28]
unifs = [unif_0, unif_1, unif_2, unif_3]

columns_right_skewed0 = [0, 2, 3, 8, 9, 13, 16]
columns_right_skewed1 = [0, 2, 3, 8, 9, 10, 13, 16, 19, 21, 23, 29]
columns_right_skewed23 =[0, 2, 5, 8, 9, 10, 13, 16, 19, 21, 23, 26, 29]
columns_right_skewed = [columns_right_skewed0, columns_right_skewed1, columns_right_skewed23, columns_right_skewed23]


sqrt_0 = [1 ,10 ,19,21]
sqrt_1 = [1, 3, 19]
sqrt_2 = [1, 3, 4, 5, 9, 19]
sqrt_3 = [1, 3, 4, 5, 6, 9, 19]
sqrt = [sqrt_0, sqrt_1, sqrt_2, sqrt_3]


num_jet0_del = [4, 5, 6, 8, 12, 22, 23, 24, 25, 26, 27, 28, 29]
num_jet1_del = [4, 5, 6, 12, 22, 26, 27, 28]
num_jet2_3_del = [22]
col_del = [num_jet0_del, num_jet1_del, num_jet2_3_del, num_jet2_3_del]


# Training phase
print("training")
ws = []  # ws for each jet
for jet_to_train in range(4):

    # Gets the rows that have jet number jet_to_train
    sep_jet_nums = seperate_PRI_jet_num(tX)
    X = tX[sep_jet_nums[jet_to_train]].copy()

    # Gets the values of the labels with jet number jet_to_train
    yt = y[sep_jet_nums[jet_to_train]].copy()

    # Clean the data
    X = clean_data(X, mean=True)
    X = classify(X, classifier[jet_to_train])
    X = unif(X, unifs[jet_to_train])
    X = apply_log(X,columns_right_skewed[jet_to_train])
    X = square_root(X, sqrt[jet_to_train])
    X = np.delete(X, col_del[jet_to_train], axis=1)
    X = standardize_data(X)

    # Feature augmentation
    sinx = np.sin(X)
    cosx = np.cos(X)

    X = np.append(X, sinx, axis=1)
    X = np.append(X, cosx, axis=1)

    X = build_poly(X, degrees[jet_to_train])

    # Train the model using ridge regression
    # and get the w corresponding to jet_to_train
    w, _ = ridge_regression(yt, X, lambdas[jet_to_train])

    ws.append(w)


# Apply our model on the test file
# load the test file
y, x, ids = load_csv_data('data/test.csv')

print("Prediction")
# Go through all the values of the jet number
for jet_to_test in range(4):

    sep_jet_nums = seperate_PRI_jet_num(x)
    X = x[sep_jet_nums[jet_to_test]].copy()

    # Clean the data and do the feature engineering
    X = clean_data(X, mean=True)
    X = classify(X, classifier[jet_to_test])
    X = unif(X, unifs[jet_to_test])
    X = apply_log(X,columns_right_skewed[jet_to_test])
    X = square_root(X, sqrt[jet_to_test])
    X = np.delete(X, col_del[jet_to_test], axis=1)
    X = standardize_data(X)

    # Feature augmentation
    sinx = np.sin(X)
    cosx = np.cos(X)

    X = np.append(X, sinx, axis=1)
    X = np.append(X, cosx, axis=1)

    X = build_poly(X, degrees[jet_to_test])

    predictions = X.dot(ws[jet_to_test])

    # Predicting the labels
    predictions[predictions >= 0] = 1
    predictions[predictions < 0] = -1

    y[sep_jet_nums[jet_to_test]] = predictions

create_csv_submission(ids, y, 'predictions.csv')

print('Results are in the file "predictions.csv"')
