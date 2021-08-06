

import numpy as np
import theano
import theano.tensor as T
from scipy.sparse import lil_matrix
import sklearn
from sklearn.preprocessing import MaxAbsScaler
import pdb
from load_data import *
from sklearn.datasets import *
from stochastic_algorithm_step_decay import *


if __name__ == '__main__':

    X_train, y_train, X_test, y_test = load_mat()
    n, d = X_train.shape
    # print(n,d)
    max_abs_scaler = MaxAbsScaler()
    X_train = max_abs_scaler.fit_transform(X_train)
    X_test = max_abs_scaler.transform(X_test)
    X_train = sklearn.preprocessing.normalize(X_train, norm='l2')
    X_test = sklearn.preprocessing.normalize(X_test, norm='l2')


    # print(X_train, y_train)
    # preprocess data
    tmp = lil_matrix((n, n))
    tmp.setdiag(y_train)
    data = theano.shared(tmp * X_train)



    l2 = 1e-4
    par = T.vector()
    loss = T.log(1 + T.exp(-T.dot(data, par))).mean() + l2 /2 * (par**2).sum()
    func = theano.function(inputs=[par], outputs=loss)
    idx = T.ivector()
    grad = theano.function(inputs=[par, idx], outputs=T.grad(loss, wrt=par),
                           givens={data: data[idx, :]})

    ##### 
    max_epoch = 128
    b = 128
    
    print('\nBegin to run SGD step decay')
    b = 128
    step_size = 10
    decay_rate = 6
    history_SGD_step_decay = sgd_step_decay(grad, X_test, y_test, step_size, decay_rate, n, d, batch=b, func=func, max_epoch=max_epoch)
    output = {'SGD step-decay: rcv1': history_SGD_step_decay}
    sio.savemat('SGD_results_step_decay_rcv1.mat', output, appendmat=True)

