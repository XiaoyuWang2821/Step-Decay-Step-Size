import theano
import theano.tensor as T
import numpy as np
import math
import time
import random
import copy
import operator

## test on TwoStage
class NeuralNetwork(object):
    def __init__(self, sizes, l2, loss='cross-entropy'):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.l2 = l2
        self.weights = [(2*np.random.rand(i, j)-1) * math.sqrt(6 / (i + j))
                        for (i, j) in zip(sizes[:-1], sizes[1:])]
        self.biases = [(2*np.random.rand(j)-1) * math.sqrt(3 / (j)) for j in sizes[1:]]
        self.parameters = self.weights + self.biases
        self.loss = loss

        # define the structure of NN
        x, y = T.matrix('x'), T.matrix('y')
        weights, biases = [], []
        a = x
        for i in range(self.num_layers - 1):
            w, b = T.matrix('w' + str(i)), T.vector('b' + str(i))
            weights.append(w)
            biases.append(b)
            a = T.dot(a, w) + b
            if i < self.num_layers - 2:
                a = T.nnet.relu(a, 0)
            if i == 0:
                norm_grad = (w**2).sum().sum()
            else:
                norm_grad += (w**2).sum().sum()
        if self.loss == 'quadratic':
            loss = ((a - y)**2).sum(axis=1).mean()
        else:
            loss = T.nnet.categorical_crossentropy(T.nnet.softmax(a), y).mean()
        obj = loss + 0.5 * self.l2 * norm_grad
        self.__loss_function__ = theano.function(inputs=[x, y]+weights+biases, outputs=[obj, loss, norm_grad])
        self.__gradient__ = theano.function(inputs=[x, y]+weights+biases, outputs=T.grad(obj, weights+biases))
        self.__predict__ = theano.function(inputs=[x]+weights+biases, outputs=T.argmax(a, axis=1))
        self.__gradient0__ = theano.function(inputs=[x, y]+weights+biases, outputs=T.grad(loss, weights+biases))



    def loss_function(self, X, Y):
        return self.__loss_function__(*([X, Y] + self.parameters))

    def gradient(self, X, Y, parameters=None):
        if parameters is None:
            g = self.__gradient__(*([X, Y] + self.parameters))
        else:
            g = self.__gradient__(*([X, Y] + parameters))
        return g

    def gradient0(self, X, Y, parameters=None):
        if parameters is None:
            g = self.__gradient0__(*([X, Y] + self.parameters))
        else:
            g = self.__gradient0__(*([X, Y] + parameters))
        return g


    def predict(self, X):
        return self.__predict__(*([X] + self.parameters))

    def accuracy(self, X, Y):
        prediction = self.predict(X)
        y = np.argmax(Y, axis=1)
        count = np.count_nonzero(prediction == y)
        return count / X.shape[0]
        


    def SGD_const(self, X, Y, X_test=None, Y_test=None, step_size_0=0.1, a0=1, num_epoch=20, m=1,batch_size=10, verbose=True):
        n = X.shape[0]
        # m = n // batch_size
        history = {'norm_square': [], 'loss': [], 'loss_test':[], 'num_grad': [], 'accuracy': []}
        TotalSFO = 0
        np.random.seed(1)
        sample = np.random.choice(n, n, replace=False)
        m = n//batch_size
        for t in range(num_epoch):
            # learning rate for this epoch
            
            step_size = step_size_0
         

            # output current status


            # SGD iterations
            for k in range(m):
                
                batch = []
                for i in range(batch_size):
                    Int_sample = (TotalSFO + i) % n
                    batch.append(sample[int(Int_sample)])
                # randomly sample a mini-batch
                #batch = random.sample(range(n), batch_size)
                TotalSFO += batch_size
                # compute the batch gradient
                grad = self.gradient(X[batch], Y[batch])
                # update iterates
                for i in range(len(self.parameters)):
                    self.parameters[i] -= step_size * grad[i]
            if verbose:
                loss = self.loss_function(X, Y)[0]
                loss_test = self.loss_function(X_test, Y_test)[0]
                if X_test is None or Y_test is None:
                    accuracy = 0
                else:
                    accuracy = self.accuracy(X_test, Y_test)
                full_grad = self.gradient(X, Y)
                norm_square = 0
                for i in range(len(self.parameters)):
                    norm_square += np.linalg.norm(full_grad[i])**2
                print('%s: Epoch: %d, Loss: %e, Loss_test: %e, Accuracy: %f, Norm^2 grad.: %e, Learning Rate: %e' %
                      (time.strftime('%H:%M:%S'), t, loss, loss_test, accuracy, norm_square, step_size))
                history['norm_square'].append(norm_square)
                history['loss'].append(loss)
                history['loss_test'].append(loss_test)
                history['num_grad'].append(TotalSFO)
                history['accuracy'].append(accuracy)

        return history

    def SGD_1t(self, X, Y, X_test=None, Y_test=None, step_size_0=0.1, a0=1, num_epoch=20, m=1,batch_size=10, verbose=True):
        n = X.shape[0]
        # m = n // batch_size
        history = {'norm_square': [], 'loss': [], 'loss_test':[], 'num_grad': [], 'accuracy': []}
        TotalSFO = 0
        np.random.seed(1)
        sample = np.random.choice(n, n, replace=False)
        m = n//batch_size
        for t in range(num_epoch):
            # learning rate for this epoch
     
            # SGD iterations
            for k in range(m):
                batch = []
                step_size = step_size_0 / (1 + a0*(t*m +k))
                for i in range(batch_size):
                    Int_sample = (TotalSFO + i) % n
                    batch.append(sample[int(Int_sample)])
                # randomly sample a mini-batch
                #batch = random.sample(range(n), batch_size)
                TotalSFO += batch_size
                # compute the batch gradient
                grad = self.gradient(X[batch], Y[batch])
                # update iterates
                for i in range(len(self.parameters)):
                    self.parameters[i] -= step_size * grad[i]
                    
            if verbose:
                loss = self.loss_function(X, Y)[0]
                loss_test = self.loss_function(X_test, Y_test)[0]
                if X_test is None or Y_test is None:
                    accuracy = 0
                else:
                    accuracy = self.accuracy(X_test, Y_test)
                full_grad = self.gradient(X, Y)
                norm_square = 0
                for i in range(len(self.parameters)):
                    norm_square += np.linalg.norm(full_grad[i])**2
                print('%s: Epoch: %d, Loss: %e, Loss_test: %e, Accuracy: %f, Norm^2 grad.: %e, Learning Rate: %e' %
                      (time.strftime('%H:%M:%S'), t, loss, loss_test, accuracy, norm_square, step_size))
                history['norm_square'].append(norm_square)
                history['loss'].append(loss)
                history['loss_test'].append(loss_test)
                history['num_grad'].append(TotalSFO)
                history['accuracy'].append(accuracy)

        return history

    def SGD_sqrt(self, X, Y, X_test=None, Y_test=None, step_size_0=0.1, a0=1, num_epoch=20, m=1,batch_size=10, verbose=True):
        n = X.shape[0]
        # m = n // batch_size
        history = {'norm_square': [], 'loss': [], 'loss_test':[], 'num_grad': [], 'accuracy': []}
        TotalSFO = 0
        np.random.seed(1)
        sample = np.random.choice(n, n, replace=False)
        m = n//batch_size
        for t in range(num_epoch):
            # learning rate for this epoch
     
            # SGD iterations
            for k in range(m):
                step_size = step_size_0 / (1 + a0*math.sqrt(t*m+k))
                batch = []
                for i in range(batch_size):
                    Int_sample = (TotalSFO + i) % n
                    batch.append(sample[int(Int_sample)])
                # randomly sample a mini-batch
                #batch = random.sample(range(n), batch_size)
                TotalSFO += batch_size
                # compute the batch gradient
                grad = self.gradient(X[batch], Y[batch])
                # update iterates
                for i in range(len(self.parameters)):
                    self.parameters[i] -= step_size * grad[i]

            # output current status
            if verbose:
                loss = self.loss_function(X, Y)[0]
                loss_test = self.loss_function(X_test, Y_test)[0]
                if X_test is None or Y_test is None:
                    accuracy = 0
                else:
                    accuracy = self.accuracy(X_test, Y_test)
                full_grad = self.gradient(X, Y)
                norm_square = 0
                for i in range(len(self.parameters)):
                    norm_square += np.linalg.norm(full_grad[i])**2
                print('%s: Epoch: %d, Loss: %e, Loss_test: %e, Accuracy: %f, Norm^2 grad.: %e, Learning Rate: %e' %
                      (time.strftime('%H:%M:%S'), t, loss, loss_test, accuracy, norm_square, step_size))
                history['norm_square'].append(norm_square)
                history['loss'].append(loss)
                history['loss_test'].append(loss_test)
                history['num_grad'].append(TotalSFO)
                history['accuracy'].append(accuracy)
                
        return history


    def SGD_exp_lr(self, X, Y, X_test=None, Y_test=None, step_size_0=0.1, beta=1, num_epoch=20, m=1,batch_size=10, verbose=True):
        n = X.shape[0]
        # m = n // batch_size
        history = {'norm_square': [], 'loss': [], 'loss_test':[], 'num_grad': [], 'accuracy': [], 'stepsize':[]}
        TotalSFO = 0
        np.random.seed(1)
        sample = np.random.choice(n, n, replace=False)
        m = n//batch_size
        max_iter = num_epoch * n // 128
        beta = beta
        decay_rate = (beta /max_iter)**(1/max_iter)
        for t in range(num_epoch):
            # learning rate for this epoch
      # output current status

            # SGD iterations
            for k in range(m):
                batch = []
                for i in range(batch_size):
                    Int_sample = (TotalSFO + i) % n
                    batch.append(sample[int(Int_sample)])
                # randomly sample a mini-batch
                #batch = random.sample(range(n), batch_size)
                TotalSFO += batch_size
                # compute the batch gradient
                grad = self.gradient(X[batch], Y[batch])
                step_size = step_size_0*decay_rate**(t*m+k)
                history['stepsize'].append(step_size)
                # update iterates
                for i in range(len(self.parameters)):
                    self.parameters[i] -= step_size * grad[i]
                    
            if verbose:
                loss = self.loss_function(X, Y)[0]
                loss_test = self.loss_function(X_test, Y_test)[0]
                if X_test is None or Y_test is None:
                    accuracy = 0
                else:
                    accuracy = self.accuracy(X_test, Y_test)
                full_grad = self.gradient(X, Y)
                norm_square = 0
                for i in range(len(self.parameters)):
                    norm_square += np.linalg.norm(full_grad[i])**2
                print('%s: Epoch: %d, Loss: %e, Loss_test: %e, Accuracy: %f, Norm^2 grad.: %e, Learning Rate: %e' %
                      (time.strftime('%H:%M:%S'), t, loss, loss_test, accuracy, norm_square, step_size))
                history['norm_square'].append(norm_square)
                history['loss'].append(loss)
                history['loss_test'].append(loss_test)
                history['num_grad'].append(TotalSFO)
                history['accuracy'].append(accuracy)
                

        return history


    def SGD_stepdecay(self, X, Y, X_test=None, Y_test=None, step_size_0=0.1, decay_rate=2, num_epoch=20, m=1,batch_size=10, verbose=True):
        n = X.shape[0]
        # m = n // batch_size
        history = {'norm_square': [], 'loss': [], 'loss_test':[], 'num_grad': [], 'accuracy': []}
        TotalSFO = 0
        np.random.seed(1)
        sample = np.random.choice(n, n, replace=False)
        m = n//batch_size
        max_iter = num_epoch * n //128
        decay_rate = decay_rate
        N_outer = math.ceil(math.log(max_iter, decay_rate)//2)
        N_inner = max_iter // N_outer
        for t in range(N_outer):
            # learning rate for this epoch
            step_size = step_size_0 / decay_rate**t

            # SGD iterations
            for k in range(N_inner):
                batch = []
                for i in range(batch_size):
                    Int_sample = (TotalSFO + i) % n
                    batch.append(sample[int(Int_sample)])
                # randomly sample a mini-batch
                #batch = random.sample(range(n), batch_size)
                TotalSFO += batch_size
                # compute the batch gradient
                grad = self.gradient(X[batch], Y[batch])
                # update iterates
                for i in range(len(self.parameters)):
                    self.parameters[i] -= step_size * grad[i]
                    

                # output current status
                if k % (N_inner // 10)== 0:
                    loss = self.loss_function(X, Y)[0]
                    loss_test = self.loss_function(X_test, Y_test)[0]
                    if X_test is None or Y_test is None:
                        accuracy = 0
                    else:
                        accuracy = self.accuracy(X_test, Y_test)
                    full_grad = self.gradient(X, Y)
                    norm_square = 0
                    for i in range(len(self.parameters)):
                        norm_square += np.linalg.norm(full_grad[i])**2
                    print('%s: Epoch: %d, Loss: %e, Loss_test: %e, Accuracy: %f, Norm^2 grad.: %e, Learning Rate: %e' %
                          (time.strftime('%H:%M:%S'), t, loss, loss_test, accuracy, norm_square, step_size))
                    history['norm_square'].append(norm_square)
                    history['loss'].append(loss)
                    history['loss_test'].append(loss_test)
                    history['num_grad'].append(TotalSFO)
                    history['accuracy'].append(accuracy)
        return history
