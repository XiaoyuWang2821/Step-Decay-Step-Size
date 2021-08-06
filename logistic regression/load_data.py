import numpy as np
import bz2
import pickle, gzip
import numpy as np
from sklearn.utils import shuffle
import scipy.io as sio

def load_MNIST():
    ''' n = 60000, d =780'''
    mnist = fetch_mldata('MNIST', data_home='.')
    X, y = mnist['data'], mnist['target']
    X = np.array(X.todense())
    train_size = 60000
    return X[:train_size], y[:train_size], X[train_size:], y[train_size:]

def load_mat():

    data ='/home/rcv1_train.binary.mat'

    file_data = sio.loadmat(data)
    X, y = file_data['Xtrain'], file_data['ytrain']
    N, d = X.shape
    train_size = int(0.75* N)

    return X[: train_size], y[: train_size], X[train_size:], y[train_size:]
