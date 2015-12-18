#! This is a simple feedfoward neural network frame , you can implement other application based on this frame.
#! Author: swind.dc.Xu
#! Email: dc.swind@gmail.com
#! Using Python2.7

import numpy as np
from Crypto.Random.random import randint
import itertools
import time
import matplotlib.pyplot as plt
from preprocess import readfile
from preprocess import readtestfile
import os
import csv

def softmax(y):
    l = len(y)
    maxv = -1.0
    maxi = -1
    for i in range(l):
        if y[i] > maxv:
            maxv = y[i]
            maxi = i
    return maxi,maxv

class Sigmoid(object):
    """ This is a logistic layer which can add no-linear capacity to FNN .
        It is always used as the last layer of the FNN. """
    def forward(self, X):
        return 1 / (1 + np.exp(-X))
    
    def backward(self, Y, T):
        """ Return the gradient of output Y. Lost = 1/2(Y - T)^2, so gY = Y - T.
            The gradient of X is y(1-y) * gY, but sometimes if y equals to 1 or 0 may lead
            the gradient to zero. So use gY instead of gX"""
        #grad = np.multiply(np.multiply(Y , (1-Y)) , (Y - T))
        grad = (Y - T)/(Y.shape[0]*Y.shape[1])# + np.sum(FNN.layer1.W)/(FNN.layer1.W.shape[0] * FNN.layer1.W.shape[1])
        #grad = grad + np.multiply(grad,T)
        return grad
    
    def cost(self, Y, T):
        return np.sum(abs(Y-T))/(Y.shape[0] * Y.shape[1])
        #return np.sum(np.multiply(Y-T,Y-T))/(Y.shape[0] * Y.shape[1])

class TanH(object):
    """TanH applies the tanh function to its inputs. I always use this layer after any linear layer. """
    def forward(self, X):
        return np.tanh(X) 
    
    def backward(self, Y, output_grad):
        """Return the gradient at the inputs of this layer."""
        #return output_grad
        gTanh = 1.0 - np.power(Y,2)
        return np.multiply(gTanh, output_grad)
        
class Linearlayer:
    def __init__(self,n_in,n_out):
        """ This random initial is recommend by ... Sorry, I forgot his name. """
        a = np.sqrt(6.0 / (n_in + n_out))
        self.W = np.random.uniform(-a, a, (n_in, n_out))
        self.b = np.zeros((n_out))
        
    def forward(self, X):
        """X is n*f, w is f * h."""
        return np.tensordot(X, self.W, axes=(-1,0)) + self.b

    def backward(self, X, gY):
        gW = np.tensordot(X, gY, axes=(0, 0))
        gB = np.sum(gY, 0)
        gX = np.tensordot(gY, self.W, axes=(-1,-1))  
        return gX, gW, gB


class FeedfowardNN:
    """Feed forward NN."""
    def __init__(self, nb_of_inputs, nb_of_outputs, nb_of_states):
        """Initialize the network layers."""
        self.layer1 = Linearlayer(nb_of_inputs, nb_of_states)  # Input layer
        self.layer2 = Linearlayer(nb_of_states, nb_of_outputs)  # Hidden layer
        self.tanh = TanH()  #no-linear function
        self.classifier = Sigmoid()  # Sigmoid output as classifier
        
    def forward(self, X):
        Z1 = self.layer1.forward(X)
        Y1 = self.tanh.forward(Z1)
        Z2 = self.layer2.forward(Y1)
        Y = self.classifier.forward(Z2) 
        return Z1, Y1, Z2, Y
     
    
    def backward(self, X, Y, Z2, Y1, Z1, T):
        gZ2 = self.classifier.backward(Y, T)
        gY1, gW2, gB2 = self.layer2.backward(Y1, gZ2)
        gZ1 = self.tanh.backward(Y1, gY1)
        gX, gW1, gB1 = self.layer1.backward(X, gZ1)
        
        return gZ2, gY1, gW2, gB2, gZ1, gX, gW1, gB1
        
    
    def getOutput(self, X):
        Z1, Y1, Z2, Y = self.forward(X)
        
        return Y  # Only return the output.
    
    def getBinaryOutput(self, X):
        return np.around(self.getOutput(X))
    
    def getParamGrads(self, X, T):
        """Return the gradients with respect to input X and target T as a list.
        The list has the same order as the get_params_iter iterator."""
        
        Z1, Y1, Z2, Y = self.forward(X)
        gZ2, gY1, gW2, gB2, gZ1, gX, gW1, gB1 = self.backward(X, Y, Z2, Y1, Z1, T)
        return [g for g in itertools.chain(
                np.nditer(gW2),
                np.nditer(gB2),
                np.nditer(gW1),
                np.nditer(gB1))]
        
        
    def cost(self, Y, T):
        """Return the cost of input X w.r.t. targets T."""
        return self.classifier.cost(Y, T)
    
    def get_params_iter(self):
        """Return an iterator over the parameters.
        The iterator has the same order as get_params_grad.
        The elements returned by the iterator are editable in-place."""
        
        return itertools.chain(
            np.nditer(self.layer2.W, op_flags=['readwrite']),
            np.nditer(self.layer2.b, op_flags=['readwrite']),
            np.nditer(self.layer1.W, op_flags=['readwrite']), 
            np.nditer(self.layer1.b, op_flags=['readwrite']))
    
class FeedfowardNN3:
    """Feed forward NN."""
    def __init__(self, nb_of_inputs, nb_of_outputs, nb_of_states):
        """Initialize the network layers."""
        self.layer1 = Linearlayer(nb_of_inputs, nb_of_states)  # Input layer
        self.layer2 = Linearlayer(nb_of_states, nb_of_states)  # Hidden layer
        self.layer3 = Linearlayer(nb_of_states, nb_of_outputs)
        self.tanh = TanH()  #no-linear function
        self.classifier = Sigmoid()  # Sigmoid output as classifier
        
    def forward(self, X):
        Z1 = self.layer1.forward(X)
        Y1 = self.tanh.forward(Z1)
        Z2 = self.layer2.forward(Y1)
        Y2 = self.tanh.forward(Z2)
        Z3 = self.layer3.forward(Y2)
        Y = self.classifier.forward(Z3) 
        return Z1, Y1, Z2, Y2, Z3, Y
     
    
    def backward(self, X, Y, Z3, Y2, Z2, Y1, Z1, T):
        gZ3 = self.classifier.backward(Y, T)
        gY2, gW3, gB3 = self.layer3.backward(Y2, gZ3)
        gZ2 = self.tanh.backward(Y2, gY2)
        gY1, gW2, gB2 = self.layer2.backward(Y1, gZ2)
        gZ1 = self.tanh.backward(Y1, gY1)
        gX, gW1, gB1 = self.layer1.backward(X, gZ1)
        
        return gZ3, gY2, gW3, gB3, gZ2, gY1, gW2, gB2, gZ1, gX, gW1, gB1
        
    
    def getOutput(self, X):
        Z1, Y1, Z2, Y2, Z3, Y = self.forward(X)
        
        return Y  # Only return the output.
    
    def getBinaryOutput(self, X):
        return np.around(self.getOutput(X))
    
    def getParamGrads(self, X, T):
        """Return the gradients with respect to input X and target T as a list.
        The list has the same order as the get_params_iter iterator."""
        
        Z1, Y1, Z2, Y2, Z3, Y = self.forward(X)
        gZ3, gY2, gW3, gB3, gZ2, gY1, gW2, gB2, gZ1, gX, gW1, gB1 = self.backward(X, Y, Z3, Y2, Z2, Y1, Z1, T)
        return [g for g in itertools.chain(
                np.nditer(gW3),
                np.nditer(gB3),
                np.nditer(gW2),
                np.nditer(gB2),
                np.nditer(gW1),
                np.nditer(gB1))]
        
        
    def cost(self, Y, T):
        """Return the cost of input X w.r.t. targets T."""
        return self.classifier.cost(Y, T)
    
    def get_params_iter(self):
        """Return an iterator over the parameters.
        The iterator has the same order as get_params_grad.
        The elements returned by the iterator are editable in-place."""
        
        return itertools.chain(
            np.nditer(self.layer3.W, op_flags=['readwrite']),
            np.nditer(self.layer3.b, op_flags=['readwrite']),
            np.nditer(self.layer2.W, op_flags=['readwrite']),
            np.nditer(self.layer2.b, op_flags=['readwrite']),
            np.nditer(self.layer1.W, op_flags=['readwrite']), 
            np.nditer(self.layer1.b, op_flags=['readwrite']))


def training(X_train,T_train,nb_train):
    """    Training    """
    # Set hyper-parameters
    lmbd = 0.5  # Rmsprop lambda
    learning_rate = 0.0001  # Learning rate
    momentum_term = 0.80  # Momentum term
    eps = 1e-12  # Numerical stability term to prevent division by zero
    mb_size = 100  # Size of the minibatches (number of samples)

    # Create the network
    nb_of_states = 100  # Number of states in the recurrent layer
    FNN = FeedfowardNN3(featureN,8,nb_of_states)

    # Set the initial parameters
    nbParameters =  sum(1 for _ in FNN.get_params_iter())  # Number of parameters in the network
    maSquare = [0.0 for _ in range(nbParameters)]  # Rmsprop moving average
    Vs = [0.0 for _ in range(nbParameters)]  # Velocity

    #test information
    cputime = time.time()
    yw = []
    ywn = 0
    for i in range(ywn):
        yw.append([])
    ye = []
    x = []
    index = 0

    # Iterate over some iterations
    for i in range(5):
        # Iterate over all the mini-batches
        for mb in range(nb_train/mb_size):
            X_mb = X_train[mb * mb_size:min((mb + 1) * mb_size,dataN-1),:]  # Input mini-batch
            T_mb = T_train[mb * mb_size:min((mb + 1) * mb_size,dataN-1),:]  # Target mini batch
            V_tmp = [v * momentum_term for v in Vs]

            # Update each parameters according to previous gradient
            for pIdx, P in enumerate(FNN.get_params_iter()):
                P += V_tmp[pIdx]
            # Get gradients after following old velocity
            backprop_grads = FNN.getParamGrads(X_mb, T_mb)  # Get the parameter gradients    
            # Update each parameter seperately
            for pIdx, P in enumerate(FNN.get_params_iter()):
                # Update the Rmsprop moving averages
                maSquare[pIdx] = lmbd * maSquare[pIdx] + (1-lmbd) * backprop_grads[pIdx]**2
                # Calculate the Rmsprop normalised gradient
                pGradNorm = learning_rate * backprop_grads[pIdx] / np.sqrt(maSquare[pIdx] + eps)
                # Update the momentum velocity
                Vs[pIdx] = V_tmp[pIdx] - pGradNorm     
                P -= pGradNorm   # Update the parameter
        

            x.append( index )
            ye.append(FNN.cost(FNN.getOutput(X_mb), T_mb ))
            for i in range(ywn):
                yw[i].append(FNN.gw3[i][0])
            index += 1
        
    print "training time",time.time() - cputime,"s"
    cputime = time.time()    
    plt.plot(x,ye)
    plt.xlabel('iterater times')
    plt.ylabel('loss')
    #plt.ylim(0,1)
    plt.figure()
    for i in range(ywn):
        plt.plot(x,yw[i])
        plt.xlabel(i)
        plt.figure("parameter"+str(i))
    return FNN



"""    Entry    """
#read file

cputime = time.time()
x,y = readfile("train.csv")
print "read train file",time.time() - cputime,"s"
cputime = time.time()
dataN = len(x)
featureN = len(x[0])
print dataN,featureN


#change 1-8 to one hot of [0,0,0,0,0,0,0,0]
Y = np.zeros((dataN,8))
for i in range(dataN):
    Y[i][y[i] - 1] = 1


X_train = x
T_train = Y
nb_train = dataN

for i in range(1):
    FNN = training(X_train,T_train,nb_train)
plt.show()    

"""    Testing    """
"""
#read test file
ID,x = readtestfile("test.csv")
print "read test file",time.time() - cputime,"s"
cputime = time.time()
csvfile = file(os.path.join(os.getcwd(), "ans.csv"),"wb")
writer = csv.writer(csvfile)
writer.writerow(["Id","Response"])
lenx = len(x)
X = np.matrix(x)
Y = FNN.getOutput(X)
for i in range(dataN):
    out = softmax(Y[i])
    writer.writerow([ID[i],out])
csvfile.close()
"""

Y = FNN.getOutput(X_train)
right = 0
count = np.zeros(8)
for i in range(dataN):
    out,v = softmax(Y[i])
    #print Y[i],out+1,y[i],v
    if out+1 == y[i]:
        right += 1
    count[out] += 1
print right,"/",dataN
print count