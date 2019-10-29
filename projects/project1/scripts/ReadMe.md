# CS-433 Machine Learning Project 1 - Finding the Higgs Boson

In this repository you'll find a few folders.
* data contains the necessary .csv files used for training and testing and is also the output folder where the pred.csv will appear
* rapport which will contain the final report PDF that was written in Overleaf
* scripts which is the main folder that contains all of the code 

In the scripts folder, there are these python files :
* implementations.py : contains the required implementations by the project of linear/ridge/logistic regression, etc...
* def.py : external function we coded that would help us throughout the project, either for cross validation or display or anything
* costs.py : file containing the code for costs computation
* helpers.py and proj1_helpers.py : two files given at the beginning containing pre-made function useful for importing and displaying infos
* project1.ipynb : the Jupyter Notebook where most tests where made
* run.py : actual file that should be run and does the prediction

## Execution 
For this project we coded everything by using Python 3.

Get into the scripts/ folder and launch
```bash
python3 run.py
```

The input are in the data folder : train.csv and test.csv

The code execute itself. We do a ridge regression on the training dataset, and a cross validation, with parameters as such :
* K-Fold : 10
* Degree : 12 up to this number to train and find the best one
* Lambda : between 10^-15 and 10^-3 
* Seed : 5 for randomness

We have also previously seperated the data to train in 4 mini-sets following a specific variable, better explained in the report.

## Submission
The output pred.csv is in the data folder and was submitter on [here](https://www.aicrowd.com/challenges/epfl-machine-learning-higgs-2019) where we achieved a score of 0.809
