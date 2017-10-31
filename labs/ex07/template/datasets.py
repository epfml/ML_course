# -*- coding: utf-8 -*-
"""provide different dataset for different usages."""
import bz2
import os
import random

import numpy as np
from scipy.sparse import csr_matrix
from sklearn import datasets

# some helper functions.


def read_sparse_vector(tokens, dimensions):
        vec = np.zeros(dimensions)
        for token in tokens:
            parts = token.split(":")
            position = int(parts[0]) - 1
            value = float(parts[1])
            vec[position] = value
        return vec


def calculate_sparsity(data):
    """data :ndarray return: float"""
    nonzeros = 0
    elements = 0
    for datapoint in data:
        nonzeros += np.count_nonzero(datapoint)
        elements += np.size(datapoint)
    return float(elements - nonzeros) / elements


def feature_scaling(data, test_data):
    feature_means = np.mean(data, axis=0)
    feature_stds = np.std(data, axis=0)
    if (feature_stds == 0).any():
        feature_stds += (feature_stds == 0)
    scaled_data = (data - feature_means) / feature_stds
    scaled_data_test = (test_data - feature_means) / feature_stds
    return scaled_data, scaled_data_test


def add_zero_feature(data):
    n, m = data.shape
#    col = np.random.randint(0,2,[n,1])*2.0-1
    col = np.ones(n, 1)
    newdata = np.append(col, data, axis=1)
    return newdata

# define basic dataset class.


class Dataset:
    directory = "datasets/"
    scaled = 0
    has_zero_feature = False

    def get_name(self):
        return self.__class__.__name__

    def get_task(self):
        pass

    def extract(self):
        """ return :(ndarray, ndarray)"""
        pass

    def extract_testdata(self):
        """ return :(ndarray, ndarray)"""
        pass

    def get_data(self):
        """ return :(ndarray, ndarray)"""
        return self.extract()

    def get_testdata(self):
        """ return :(ndarray, ndarray)"""
        return self.extract_testdata()

# define dataset based on task.


class ClassificationDataset(Dataset):
    def get_task(self):
        return "classification"


class RegressionDataset(Dataset):
    def get_task(self):
        return "regression"


class RecommendationDataset(Dataset):
    def get_task(self):
        return "recommendation"


# define the data type of the file.


class SparseDataFile(Dataset):

    def extract_line(self, line):
        segments = line.split()
        cls = self.get_label(segments[0])
        vec = read_sparse_vector(segments[1:], self.number_of_features)
        return vec, cls

    def get_label(self, labelstr):
        # should be 0,1
        pass


class ASCIIfile(Dataset):
    def read_label_file(self, file):
        labels = list()
        for l in file.readlines():
            labels.append(self.get_label(l.strip()))
        return labels

    def get_label(self, labelstr):
        pass

    def extract_line(self, line):
        values = line.strip().split(' ')
        value_list = list()
        for v in values:
            value_list.append(float(v))
        return np.array(value_list)


# define various datasets.


class Covtype(ClassificationDataset, SparseDataFile):
    base = "http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets"
    url = base + "/binary/covtype.libsvm.binary.bz2"
    filename = "covtype.libsvm.binary.bz2"

    number_of_points = 581012
    number_of_features = 54

    def get_label(self, labelstr):
        return float(labelstr) - 1.0

    def extract(self):
        path = self.directory + self.filename
        f = bz2.BZ2File(path, 'r')
        datapoints = []
        labels = []
        for (index, line) in enumerate(f.readlines()):
            if line.isspace():
                continue
            line = line.decode()
            vec, cls = self.extract_line(line.strip())
            datapoints.append(vec)
            labels.append(cls)
        return np.array(datapoints), np.array(labels)

    def extract_testdata(self):
        return self.extract()


class ijcnn1(ClassificationDataset, SparseDataFile):
    base = "http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets"
    url = base + "/binary/covtype.libsvm.binary.bz2"
    filename = "ijcnn1.bz2"
    filename_test = "ijcnn1.t.bz2"

    number_of_points = 49990
    number_of_features = 22

    def get_label(self, labelstr):
        return (float(labelstr) + 1.0)/2.0

    def extract(self):
        path = self.directory + self.filename
        f = bz2.BZ2File(path, 'r')
        datapoints = []
        labels = []
        for (index, line) in enumerate(f.readlines()):
            if line.isspace():
                continue
            line = line.decode()
            vec, cls = self.extract_line(line.strip())
            datapoints.append(vec)
            labels.append(cls)
        return np.array(datapoints), np.array(labels)

    def extract_testdata(self):
        path = self.directory + self.filename_test
        f = bz2.BZ2File(path, 'r')
        datapoints = []
        labels = []
        for (index, line) in enumerate(f.readlines()):
            if line.isspace():
                continue
            line = line.decode()
            vec, cls = self.extract_line(line.strip())
            datapoints.append(vec)
            labels.append(cls)
        return np.array(datapoints), np.array(labels)


class RCV1(ClassificationDataset, SparseDataFile):
    filename = "rcv1_train.binary.bz2"
    filename_test = "rcv1_test.binary.bz2"

    number_of_points = 20242
    number_of_points_test = 677399
    number_of_features = 47236

    def get_label(self, labelstr):
        return float(labelstr)

    def extract(self):
        path = self.directory + self.filename
        f = bz2.BZ2File(path, 'r')
        datapoints = []
        labels = []
        for (index, line) in enumerate(f.readlines()):
            if line.isspace():
                continue
            line = line.decode()
            vec, cls = self.extract_line(line.strip())
            datapoints.append(vec)
            labels.append(cls)
        return np.array(datapoints), np.array(labels)

    def extract_testdata(self):
        return self.extract()


class A8A(ClassificationDataset):
    base = "http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets"
    url = base + "/binary/rcv1_train.binary.bz2"
    filename = "a8a.txt"
    filename_test = "a8a.t.txt"

    def extract(self):
        number_of_points = 22696
        number_of_features = 123
        path = self.directory + self.filename
        f = open(path, 'r')

        data = np.zeros((number_of_points, number_of_features))
        labels = np.zeros((number_of_points))

        for (index, line) in enumerate(f.readlines()):
            line.strip()
            splitted = line.split()

            # y \in {0,1}
            labels[index] = (int(splitted[0]) + 1) / 2
            for element in splitted[1:]:
                j = element.split(':')
                data[index, int(j[0])-1] = float(j[1])

        f.close()
        return (data, labels)

    def extract_testdata(self):
        number_of_points = 9865
        number_of_features = 123
        path = self.directory + self.filename_test
        f = open(path, 'r')

        data = np.zeros((number_of_points, number_of_features))
        labels = np.zeros((number_of_points))

        for (index, line) in enumerate(f.readlines()):
            line.strip()
            splitted = line.split()
            # y \in {0,1}
            labels[index] = (int(splitted[0])+1)/2
            for element in splitted[1:]:
                j = element.split(':')
                data[index, int(j[0])-1] = float(j[1])

        f.close()
        return (data, labels)


class Mushroom(ClassificationDataset):
    base = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets"
    url = base + "/binary/mushrooms"
    filename = "mushrooms.txt"

    def extract(self):
        number_of_points = 8124
        number_of_features = 112
        path = self.directory + self.filename
        f = open(path, 'r')

        data = np.zeros((number_of_points, number_of_features))
        labels = np.zeros((number_of_points))

        for (index, line) in enumerate(f.readlines()):
            line.strip()
            splitted = line.split()
            # y \in {0,1}
            labels[index] = (int(splitted[0])-1)
            for element in splitted[1:]:
                j = element.split(':')
                data[index, int(j[0])-1] = float(j[1])

        f.close()
        return (data, labels)

    def extract_testdata(self):
        return self.extract()


class MNIST(ClassificationDataset):
    base = "http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets"
    url = base + "/multiclass/mnist.bz2"
    filename = "mnist.bz2"

    number_of_points = 60000
    number_of_features = 780

    def get_label(self, labelstr):
        # -1.0
        return float(labelstr)

    def extract(self):
        path = self.directory + self.filename
        f = bz2.BZ2File(path, 'r')
        datapoints = []
        labels = []
        for (index, line) in enumerate(f.readlines()):
            if line.isspace():
                continue
            line = line.decode()
            segments = line.strip().split()

            cls = int(segments[0])
#            cls =  0.0 if int(segments[0])<=4 else 1.0
            vec = read_sparse_vector(segments[1:], self.number_of_features)

            datapoints.append(vec)
            labels.append(cls)

        return np.array(datapoints), np.array(labels)

    def extract_testdata(self):
        return self.extract()


class GISETTE(ClassificationDataset, SparseDataFile):
    base = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets"
    url = base + "/binary/webspam_wc_normalized_trigram.svm.bz2"
    filename = "gisette_scale.bz2"
    filename_test = "gisette_scale.t.bz2"
    scaled = 1

    def get_label(self, labelstr):
        return (float(labelstr)+1)/2

    def extract(self):
        path = self.directory + self.filename
        self.number_of_points = 6000
        self.number_of_features = 5000
        f = bz2.BZ2File(path, 'r')
        datapoints = []
        labels = []
        for (index, line) in enumerate(f.readlines()):
            if line.isspace():
                continue
            line = line.decode()
            vec, cls = self.extract_line(line.strip())
            datapoints.append(vec)
            labels.append(cls)
        return np.array(datapoints), np.array(labels)

    def extract_testdata(self):
        path = self.directory + self.filename_test
        self.number_of_points = 1000
        self.number_of_features = 5000
        f = bz2.BZ2File(path, 'r')
        datapoints = []
        labels = []
        for (index, line) in enumerate(f.readlines()):
            if line.isspace():
                continue
            line = line.decode()
            vec, cls = self.extract_line(line.strip())
            datapoints.append(vec)
            labels.append(cls)
        return np.array(datapoints), np.array(labels)
