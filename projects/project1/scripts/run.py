import numpy as np

from implementations import *
from helpers import *
from defs import *
from proj1_helpers import *



##############################################################
print("STARTING")
'''Data for training shaping'''

DATA_TRAIN_PATH = '../data/train.csv'


y0, tX0, id0 = load_data_jet_number(DATA_TRAIN_PATH, 0, sub_sample=False)
y1, tX1, id1 = load_data_jet_number(DATA_TRAIN_PATH, 1, sub_sample=False)
y2, tX2, id2 = load_data_jet_number(DATA_TRAIN_PATH, 2, sub_sample=False)
y3, tX3, id3 = load_data_jet_number(DATA_TRAIN_PATH, 3, sub_sample=False)

for col in [0,1,2]:
    tX0[tX0[:,col]==-999,col]=0
    tX1[tX1[:,col]==-999,col]=0
    tX2[tX2[:,col]==-999,col]=0
    tX3[tX3[:,col]==-999,col]=0

tX0_standardized, tX0_mean, tX0_std= standardize(tX0)
tX1_standardized, tX1_mean, tX1_std= standardize(tX1)
tX2_standardized, tX2_mean, tX2_std= standardize(tX2)
tX3_standardized, tX3_mean, tX3_std= standardize(tX3)


'''Data training'''

k_fold=4
seed=5
degree=8
lower_lambda=-10
upper_lambda=0
print("FIRST MATRIX")
weights_0, loss_0, deg0 = cross_validation_best_weight(y0, tX0_standardized, k_fold, degree, seed, lower_lambda, upper_lambda, "jet0")
print("SECOND MATRIX")
weights_1, loss_1, deg1 = cross_validation_best_weight(y1, tX1_standardized, k_fold, degree, seed, lower_lambda, upper_lambda, "jet1")
print("THIRD MATRIX")
weights_2, loss_2, deg2 = cross_validation_best_weight(y2, tX2_standardized, k_fold, degree, seed, lower_lambda, upper_lambda, "jet2")
print("FORTH MATRIX")
weights_3, loss_3, deg3 = cross_validation_best_weight(y3, tX3_standardized, k_fold, degree, seed, lower_lambda, upper_lambda, "jet3")


'''Data for testing shaping'''

DATA_TEST_PATH = '../data/test.csv' 

_, tX0_te, id0_te = load_data_jet_number(DATA_TEST_PATH, 0)
_, tX1_te, id1_te = load_data_jet_number(DATA_TEST_PATH, 1)
_, tX2_te, id2_te = load_data_jet_number(DATA_TEST_PATH, 2)
_, tX3_te, id3_te = load_data_jet_number(DATA_TEST_PATH, 3)

for col in [0,1,2]:
    tX0_te[tX0_te[:,col]==-999,col]=0
    tX1_te[tX1_te[:,col]==-999,col]=0
    tX2_te[tX2_te[:,col]==-999,col]=0
    tX3_te[tX3_te[:,col]==-999,col]=0

tX0_te_standardized, tX0_te_mean, tX0_te_std= standardize(tX0_te)
tX1_te_standardized, tX1_te_mean, tX1_te_std= standardize(tX1_te)
tX2_te_standardized, tX2_te_mean, tX2_te_std= standardize(tX2_te)
tX3_te_standardized, tX3_te_mean, tX3_te_std= standardize(tX3_te)


'''Data testing'''

OUTPUT_PATH = '../data/pred.csv' 
y0_pred = predict_labels(weights_0, build_poly(tX0_te_standardized,deg0))
y1_pred = predict_labels(weights_1, build_poly(tX1_te_standardized,deg1))
y2_pred = predict_labels(weights_2, build_poly(tX2_te_standardized,deg2))
y3_pred = predict_labels(weights_3, build_poly(tX3_te_standardized,deg3))

y_pred_all=np.concatenate([y0_pred,y1_pred,y2_pred,y3_pred])
ids_pred_all=np.concatenate([id0_te,id1_te,id2_te,id3_te])
y_and_ids=list(zip(*sorted(zip(ids_pred_all, y_pred_all))))
ids_test=y_and_ids[0]
y_pred=y_and_ids[1]

create_csv_submission(ids_test, y_pred, OUTPUT_PATH)
print("PREDICTION DONE")
