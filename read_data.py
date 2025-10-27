import numpy as np
from numpy.random import default_rng

#read dataset, returns data inputs, labels and unique class names
def read_dataset(filepath):
    data = np.loadtxt(filepath)
    x = data[:, :-1]
    y_labels = data[:, -1]
    [classes, y] = np.unique(y_labels, return_inverse=True)
    return (x, y, classes)

#splits dataset into training and testing sets
def split_dataset(x, y, test_proportion, random_generator=default_rng()):
    shuffled_indices = random_generator.permutation(len(x))
    n_test = round(len(x) * test_proportion)
    n_train = len(x) - n_test
    x_train = x[shuffled_indices[:n_train]]
    y_train = y[shuffled_indices[:n_train]]
    x_test = x[shuffled_indices[n_train:]]
    y_test = y[shuffled_indices[n_train:]]
    return (x_train, x_test, y_train, y_test)