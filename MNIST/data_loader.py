import numpy as np
from sklearn.datasets import fetch_mldata
import pickle
import pickle, gzip
import numpy as np
from sklearn.utils import shuffle

def load_CIFAR10(dir='cifar-10-batches-py/'):
    # load data from files
    for i in range(1, 6):
        batch = pickle.load(open(dir + 'data_batch_' + str(i), 'rb'), encoding='latin1')
        if i == 1:
            X_train, y_train = batch['data'], np.array(batch['labels'])
        else:
            X_train = np.vstack((X_train, batch['data']))
            y_train = np.hstack((y_train, np.array(batch['labels'])))
    batch = pickle.load(open(dir + 'test_batch', 'rb'), encoding='latin1')
    X_test, y_test = batch['data'], np.array(batch['labels'])
    return X_train, y_train, X_test, y_test


def load_MNIST():
    ''' n = 60000, d =780'''
    mnist = fetch_mldata('MNIST', data_home='.')
    X, y = mnist['data'], mnist['target']
    X = np.array(X.todense())
    train_size = 60000
    return X[:train_size], y[:train_size], X[train_size:], y[train_size:]

def load_MNIST2():
    '''the two mnist dataset are similar, in this case
    n_train = 60000, d = 784, n_test = 10000'''
    f = gzip.open('mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
    X_train = np.concatenate((train_set[0], valid_set[0]), axis=0)
    y_train = np.concatenate((train_set[1], valid_set[1]), axis=0)
    X_test = test_set[0]
    y_test = test_set[1]
    X_train, y_train = shuffle(X_train, y_train, random_state=0)
    X_test, y_test = shuffle(X_test, y_test, random_state=0)
    return X_train, y_train, X_test, y_test

