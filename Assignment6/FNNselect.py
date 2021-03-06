#! This is a simple feedfoward neural network frame , you can implement other application based on this frame.
#! Author: swind.dc.Xu
#! Email: dc.swind@gmail.com
#! Using Python2.7

import numpy as np
from Crypto.Random.random import randint
import random
import itertools
import time
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig
from preprocessv2 import process
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
    def __init__(self,n_in,n_out,dropout = 0):
        """ This random initial is recommend by ... Sorry, I forgot his name. """
        a = np.sqrt(6.0 / (n_in + n_out))
        self.W = np.random.uniform(-a, a, (n_in, n_out))
        self.b = np.zeros((n_out))
        self.n_in = n_in
        self.n_out = n_out
        self.dropout = dropout
        
    def forward(self, X):
        """X is n*f, w is f * h."""
        self.Drop = np.round(np.random.uniform(0,1,self.n_out))
        W = self.W
        b = self.b
        if self.dropout == 1:
            W = np.multiply(self.W,self.Drop)
            b = np.multiply(self.b,self.Drop)
        return np.tensordot(X, W, axes=(-1,0)) + b

    def backward(self, X, gY):
        gW = np.tensordot(X, gY, axes=(0, 0))
        gB = np.sum(gY, 0)
        gX = np.tensordot(gY, self.W, axes=(-1,-1))  
        if self.dropout == 1:
            gW = np.multiply(gW, self.Drop)
            gB = np.multiply(gB, self.Drop)
            gX = np.tensordot(gY, np.multiply(self.W,self.Drop), axes=(-1,-1))
        return gX, gW, gB
    
class FeedfowardNN3:
    """Feed forward NN."""
    def __init__(self, nb_of_inputs, nb_of_outputs, nb_of_states, dropout = 0):
        """Initialize the network layers."""
        self.layer1 = Linearlayer(nb_of_inputs, nb_of_states, dropout)  # Input layer
        self.layer2 = Linearlayer(nb_of_states, nb_of_states/2, dropout)  # Hidden layer
        self.layer3 = Linearlayer(nb_of_states/2, nb_of_outputs)
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
        return [gW3,gB3,gW2,gB2,gW1,gB1]
        
        
    def cost(self, Y, T):
        """Return the cost of input X w.r.t. targets T."""
        return self.classifier.cost(Y, T)
    
    def get_params_iter(self):
        """Return an iterator over the parameters.
        The iterator has the same order as get_params_grad.
        The elements returned by the iterator are editable in-place."""
        return [self.layer3.W, self.layer3.b, self.layer2.W, self.layer2.b, self.layer1.W, self.layer1.b]

def minibatch_sample(X_train, T_train, mb_size):
    samples = np.sort(random.sample(range(0,X_train.shape[0]),mb_size))
    X_mb = np.zeros_like(X_train[0:mb_size])
    T_mb = np.zeros_like(T_train[0:mb_size])
    for i in range(mb_size):
        X_mb[i] = X_train[samples[i]]
        T_mb[i] = T_train[samples[i]]
    return X_mb, T_mb

def training(X_train,T_train,nb_train,mb,regu,nb,dr):
    """    Training    """
    # Set hyper-parameters
    lmbd = 0.5  # Rmsprop lambda
    learning_rate = 0.0001  # Learning rate
    momentum_term = 0.80  # Momentum term
    eps = 1e-12  # Numerical stability term to prevent division by zero
    mb_size = mb  # Size of the minibatches (number of samples)
    regular_lamda = regu

    # Create the network
    nb_of_states = nb  # Number of states in the recurrent layer
    FNN = FeedfowardNN3(featureN,8,nb_of_states,dr)
    outputfilename = "ans_mb"+str(mb_size)+"_regu"+str(regular_lamda)+"_state"+str(nb_of_states)+"_dr"+str(dr)+".csv"
    
    # Set the initial parameters
    maSquare = [np.zeros_like(param) for param in FNN.get_params_iter()]  # Rmsprop moving average
    Vs = [np.zeros_like(param) for param in FNN.get_params_iter()]  # Velocity
    
    #test information
    cputime = time.time()
    ye = []
    x = []
    index = 0

    # Iterate over some iterations
    for i in range(5):
        print i+1,"th epoch",
        # Iterate over all the mini-batches
        for mb in range(nb_train/mb_size):
            X_mb , T_mb = minibatch_sample(X_train, T_train, mb_size)
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
                pGradNorm = learning_rate * (backprop_grads[pIdx] +  regular_lamda * P/nb_train)/ np.sqrt(maSquare[pIdx] + eps)
                # Update the momentum velocity
                Vs[pIdx] = V_tmp[pIdx] - pGradNorm     
                P -= pGradNorm   # Update the parameter
        
            x.append( index )
            ye.append(FNN.cost(FNN.getOutput(X_mb), T_mb ))
            index += 1

        print "    training time",time.time() - cputime,"s"
        cputime = time.time()
    
    plt.plot(x, ye)
    plt.xlabel("iterater times")
    plt.ylabel("loss")
    #plt.show()
    savefig("ans/"+outputfilename+".jpg")
    return FNN,outputfilename



"""    Entry    """
#read file
pr = process()
x,y = pr.readtrainfile("train.csv")
dataN = x.shape[0]
featureN = x.shape[1]
print "data",dataN,"feature",featureN

#change 1-8 to one hot of [0,0,0,0,0,0,0,0]
Y = np.zeros((dataN,8))
for i in range(dataN):
    Y[i][y[i] - 1] = 1

X_train = x
T_train = Y
nb_train = dataN

mb = [64,64]
regu = [5.0,0.0]
nb = [128,128]
dr = [0,1]

for i in range(len(mb)):
    runninglog = open('log.txt','a+')
    FNN, filename = training(X_train,T_train,nb_train,mb[i],regu[i],nb[i],dr[i])
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
    print np.around(count)
    runninglog.write(filename+"\n")
    runninglog.write(str(right)+"/"+str(dataN)+"\n")
    for i in range(8):
        runninglog.write(str(round(count[i]))+" ")
    runninglog.write("\n"+"---------------------------------"+"\n")
    
    """    Testing    """

    #read test file
    ID,x = pr.readtestfile("test.csv")
    csvfile = file(os.path.join(os.getcwd(), "ans/"+filename),"wb")
    writer = csv.writer(csvfile)
    writer.writerow(["Id","Response"])
    lenx = x.shape[0]
    X = x
    Y = FNN.getOutput(X)
    for i in range(lenx):
        out,v = softmax(Y[i])
        writer.writerow([ID[i],out+1])
    csvfile.close()
    runninglog.close()
