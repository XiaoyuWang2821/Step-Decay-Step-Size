import numpy as np
from sklearn.preprocessing import MinMaxScaler
from nn_theano_relu_stepdecay import NeuralNetwork
import multiprocessing  as mp
from data_loader import *
from scipy.io import savemat
import pickle
import copy
from itertools import product
import os


# load MINIST dataset
X_train, y_train, X_test, y_test = load_MNIST2()


min_max_scaler = MinMaxScaler()
X_train = min_max_scaler.fit_transform(X_train)
X_test = min_max_scaler.transform(X_test)

# size of training data
n, d = X_train.shape
# number of classes
num_class = 10

# transform y into a matrix, where each row has one 1 and others 0
Y_train = np.zeros((y_train.shape[0], num_class))
Y_train[range(y_train.shape[0]), y_train] = 1.0

Y_test = np.zeros((y_test.shape[0], num_class))
Y_test[range(y_test.shape[0]), y_test] = 1.0



decay_rate = 6
step_size_0 = 0.5

NN = NeuralNetwork([d, 100, num_class], 1e-4, loss='cross-entropy')

print('\n Begin to run SGD: stepdecay')

history = NN.SGD_stepdecay(X_train, Y_train, X_test=X_test, Y_test=Y_test, step_size_0=step_size_0, decay_rate=decay_rate, num_epoch=128, m=n//128, batch_size=128)
output = {'sgd stepdecay': history}
savemat('sgd_mnist_stepdecay.mat', output, appendmat=True)

