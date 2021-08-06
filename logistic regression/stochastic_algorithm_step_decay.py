#!/usr/bin/env python

import numpy as np
import random
import time
import math


# sgd_const is SGD with constant step-size;
# sgd_sqrt is  SGD with \eta_t = \eta_0/(1+a\sqrt{t})
# sgd_explr is SGD with the step-size which exponentially decays after some iterates (see Hazan & kale 2014).
# sgd_exp_rate is SGD with exponential decay step-size proposed by Li et al. 2021
# sgd_step_decay is SGD with step-decay step-size

def accuracy(X_test, y_test, x):
    n = X_test.shape[0]
    y_predict = np.sign(np.dot(X_test, x))
    count = np.count_nonzero(y_test == y_predict) * 1.0
    accu = count/n
    return  accu



def sgd(grad, X_test, y_test, step_size, n, d, batch=128, max_epoch=128, x0=None, a0=1, beta=None,
           func=None, verbose=True):

    if x0 is None:
        x = np.zeros(d)
    elif isinstance(x0, np.ndarray) and x0.shape == (d, ):
        x = x0.copy()
    else:
        raise ValueError('x0 must be a numpy array of size (d, )')

    n_test = X_test.shape[0]
    history = {'accuracy': [], 'func': [], 'norm_square': [], 'num_grad': [], 'step_size': step_size
               }
    step_size_0 = step_size
    np.random.seed(1)
    
    for k in range(max_epoch):
        m = n // 128
        #step_size = step_size_0 / (1 + a0*k)
  


        for i in range(m):
            step_size = step_size_0/ (1+ a0*(k*m+i))
            idx = (random.randrange(n), batch)
            g = grad(x, idx)
            x -= step_size * g
            
            
        if verbose:
            full_grad = grad(x, range(n))
            y_predict = np.sign(np.dot(X_test, x))
            accu = np.count_nonzero(np.transpose(y_test) == y_predict) * 1.0 / n_test
            norm_square = np.linalg.norm(full_grad)**2
            func_value = func(x)
            output = 'Epoch.: %d, Step size: %.2e, Grad. norm: %e, Accuracy.: %f' % \
                     (k, step_size, norm_square, accu)
            if func is not None:
                output += ', Func. value: %e' % func_value
            print(output)
            history['accuracy'].append(accu)
            history['func'].append(func_value)
            history['norm_square'].append(norm_square)
            history['num_grad'].append((k+1) * m * batch)


    return history


def sgd_const(grad, X_test, y_test, step_size, n, d, batch=128, max_epoch=100, x0=None, a0=1, beta=None,
           func=None, verbose=True):
    """
    
        grad: gradient function in the form of grad(x, idx), where idx is a list of induces
        init_step_size: initial step size
        n, d: size of the problem
        m: step sie updating frequency
        beta: the averaging parameter
        phi: the smoothing function in the form of phi(k)
        func: the full function, f(x) returning the function value at x
    """

    if x0 is None:
        x = np.zeros(d)
    elif isinstance(x0, np.ndarray) and x0.shape == (d, ):
        x = x0.copy()
    else:
        raise ValueError('x0 must be a numpy array of size (d, )')

    n_test = X_test.shape[0]
    history = {'accuracy': [], 'func': [], 'norm_square': [], 'num_grad': [], 'step_size': step_size
               }
    step_size_0 = step_size
    np.random.seed(1)
    for k in range(max_epoch):
        m = n // 128
        step_size = step_size_0


        for i in range(m):
            # step_size = step_size_0 / (1 + k*m +i)
            idx = (random.randrange(n), batch)
            g = grad(x, idx)
            x -= step_size * g
            
        if verbose:
            full_grad = grad(x, range(n))
            y_predict = np.sign(np.dot(X_test, x))
            accu = np.count_nonzero(np.transpose(y_test) == y_predict) * 1.0 / n_test
            norm_square = np.linalg.norm(full_grad)**2
            func_value = func(x)
            output = 'Epoch.: %d, Step size: %.2e, Grad. norm: %e, Accuracy.: %f' % \
                     (k, step_size, norm_square, accu)
            if func is not None:
                output += ', Func. value: %e' % func_value
            print(output)
            history['accuracy'].append(accu)
            history['func'].append(func_value)
            history['norm_square'].append(norm_square)
            history['num_grad'].append((k+1) * m * batch)


    return history

def sgd_sqrt(grad, X_test, y_test, step_size, n, d, batch=100, max_epoch=100, x0=None, a0=1, beta=None,
           func=None, verbose=True):


    if x0 is None:
        x = np.zeros(d)
    elif isinstance(x0, np.ndarray) and x0.shape == (d, ):
        x = x0.copy()
    else:
        raise ValueError('x0 must be a numpy array of size (d, )')

    n_test = X_test.shape[0]
    history = {'accuracy': [], 'func': [], 'norm_square': [], 'num_grad': [], 'step_size': step_size
               }
    step_size_0 = step_size
    np.random.seed(1)

    for k in range(max_epoch):
        m = n//128
        #step_size = step_size_0


        for i in range(m):
            step_size = step_size_0 / (1 + a0*math.sqrt(k*m +i))
            idx = (random.randrange(n), batch)
            g = grad(x, idx)
            x -= step_size * g


        if verbose:
            full_grad = grad(x, range(n))
            y_predict = np.sign(np.dot(X_test, x))
            accu = np.count_nonzero(np.transpose(y_test) == y_predict) * 1.0 / n_test
            norm_square = np.linalg.norm(full_grad)**2
            func_value = func(x)
            output = 'Epoch.: %d, Step size: %.2e, Grad. norm: %e, Accuracy.: %f' % \
                     (k, step_size, norm_square, accu)
            if func is not None:
                output += ', Func. value: %e' % func_value
            print(output)
            history['accuracy'].append(accu)
            history['func'].append(func_value)
            history['norm_square'].append(norm_square)
            history['num_grad'].append((k+1) * m * batch)

    return history


def sgd_exp_lr(grad, X_test, y_test, step_size, n, d, T_0, batch=100, max_epoch=100, x0=None, a0=1, beta=None,
           func=None, verbose=True):

    if x0 is None:
        x = np.zeros(d)
    elif isinstance(x0, np.ndarray) and x0.shape == (d, ):
        x = x0.copy()
    else:
        raise ValueError('x0 must be a numpy array of size (d, )')

    n_test = X_test.shape[0]
    history = {'accuracy': [], 'func': [], 'norm_square': [], 'num_grad': [], 'step_size': step_size
               }
    step_size_0 = step_size
    np.random.seed(1)
    T_0 = T_0
    
    max_iter = max_epoch * n //128
    T_1 = 0
    T_hat = T_2 = T_0
    T_num = 0
    k = 0
    while T_num <= max_iter:
        #m = n//100
        #print(T_num)
        #step_size = step_size_0


        #print(1)
        for i in range(T_2-T_1):
            #print(2)
            # step_size = step_size_0 / (1 + k*m +i)
            idx = (random.randrange(n), batch)
            g = grad(x, idx)
            x -= step_size * g
            
            ### print
            if i % (T_2-T_1)//4 == 0 :
                full_grad = grad(x, range(n))
                y_predict = np.sign(np.dot(X_test, x))
                accu = np.count_nonzero(np.transpose(y_test) == y_predict) * 1.0 / n_test
                norm_square = np.linalg.norm(full_grad)**2
                func_value = func(x)
                output = 'Epoch.: %d, Step size: %.2e, Grad. norm: %e, Accuracy.: %f' % \
                         (k, step_size, norm_square, accu)
                if func is not None:
                    output += ', Func. value: %e' % func_value
                print(output)
                history['accuracy'].append(accu)
                history['func'].append(func_value)
                history['norm_square'].append(norm_square)
                history['num_grad'].append(T_num * batch + i*batch)
        #print(T_2 - T_1)
        T_num = T_num + T_2 - T_1
        T_hat = 2*T_2
        T_1= T_2
        T_2 += T_hat
        step_size = step_size /2
        k += 1
        #print(T_num)
    else:
        full_grad = grad(x, range(n))
        y_predict = np.sign(np.dot(X_test, x))
        accu = np.count_nonzero(np.transpose(y_test) == y_predict) * 1.0 / n_test
        norm_square = np.linalg.norm(full_grad)**2
        func_value = func(x)
        output = 'Epoch.: %d, Step size: %.2e, Grad. norm: %e, Accuracy.: %f' % \
                 (k, step_size, norm_square, accu)
        if func is not None:
            output += ', Func. value: %e' % func_value
        print(output)
        history['accuracy'].append(accu)
        history['func'].append(func_value)
        history['norm_square'].append(norm_square)
        history['num_grad'].append(T_num * batch)
        print('all done')
        
    return history

def sgd_exp_rate(grad, X_test, y_test, step_size, beta, n, d, batch=128, max_epoch=100, x0=None,
           func=None, verbose=True):


    if x0 is None:
        x = np.zeros(d)
    elif isinstance(x0, np.ndarray) and x0.shape == (d, ):
        x = x0.copy()
    else:
        raise ValueError('x0 must be a numpy array of size (d, )')

    n_test = X_test.shape[0]
    history = {'accuracy': [], 'func': [], 'norm_square': [], 'num_grad': [], 'step_size': step_size
               }
    step_size_0 = step_size
    max_iter = max_epoch * n//batch
    beta = beta
    rate = (beta /max_iter)**(1/max_iter)
    np.random.seed(1)
    for k in range(max_epoch):
        m =n //batch
        #step_size = step_size_0


        for i in range(m):
            step_size = step_size_0 * rate**(m*k + i)
            idx = (random.randrange(n), batch)
            g = grad(x, idx)
            x -= step_size * g
            
        if verbose:
            full_grad = grad(x, range(n))
            y_predict = np.sign(np.dot(X_test, x))
            accu = np.count_nonzero(np.transpose(y_test) == y_predict) * 1.0 / n_test
            norm_square = np.linalg.norm(full_grad)**2
            func_value = func(x)
            output = 'Epoch.: %d, Step size: %.2e, Grad. norm: %e, Accuracy.: %f' % \
                     (k, step_size, norm_square, accu)
            if func is not None:
                output += ', Func. value: %e' % func_value
            print(output)
            history['accuracy'].append(accu)
            history['func'].append(func_value)
            history['norm_square'].append(norm_square)
            history['num_grad'].append((k+1)* m * batch)


    return history





def sgd_step_decay(grad, X_test, y_test, step_size, decay_rate, n, d, batch=128, max_epoch=128, x0=None, beta=None,
           func=None, verbose=True):


    if x0 is None:
        x = np.zeros(d)
    elif isinstance(x0, np.ndarray) and x0.shape == (d, ):
        x = x0.copy()
    else:
        raise ValueError('x0 must be a numpy array of size (d, )')

    n_test = X_test.shape[0]
    history = {'accuracy': [], 'func': [], 'norm_square': [], 'num_grad': [], 'step_size': step_size
               }
    step_size_0 = step_size
    np.random.seed(1)
    max_iter = max_epoch * n //batch
    alpha = decay_rate
    N_outer = np.int(math.log(max_iter, alpha))
    S_inner = max_iter // N_outer
   
    for k in range(N_outer):
        
        step_size = step_size_0 / alpha**k
 
        for i in range(S_inner):
            idx = (random.randrange(n), batch)
            g = grad(x, idx)
            x -= step_size * g

            if i % int(S_inner/10) == 0:
                full_grad = grad(x, range(n))
                y_predict = np.sign(np.dot(X_test, x))
                accu = np.count_nonzero(np.transpose(y_test) == y_predict) * 1.0 / n_test
                norm_square = np.linalg.norm(full_grad)**2
                func_value = func(x)
                output = 'Epoch.: %d, Step size: %.2e, Grad. norm: %e, Accuracy.: %f' % \
                         (k, step_size, norm_square, accu)
                if func is not None:
                    output += ', Func. value: %e' % func_value
                print(output)
                history['accuracy'].append(accu)
                history['func'].append(func_value)
                history['norm_square'].append(norm_square)
                history['num_grad'].append(k * S_inner * batch + i*batch)

    return history

