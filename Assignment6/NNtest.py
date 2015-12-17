import numpy as np
from Crypto.Random.random import randint
import itertools
import time
import matplotlib.pyplot as plt

class LogisticClassifier(object):
    def forward(self, X):
        return 1 / (1 + np.exp(-X))
    
    def backward(self, Y, T):
        #grad = np.multiply(np.multiply(Y , (1-Y)) , (Y - T))
        grad = (Y - T)/Y.shape[0]
        return grad
    
    def cost(self, Y, T):
        return np.sum(abs(Y-T))/Y.shape[0]

class Linearlayer:
    def __init__(self,n_in,n_out):
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

class TanH(object):
    """TanH applies the tanh function to its inputs."""
    def forward(self, X):
        return np.tanh(X) 
    
    def backward(self, Y, output_grad):
        """Return the gradient at the inputs of this layer."""
        gTanh = 1.0 - np.power(Y,2)
        return np.multiply(gTanh, output_grad)
    
class FeedfowardNN:
    """Feedfoward NN."""
    def __init__(self, nb_of_inputs, nb_of_outputs, nb_of_states):
        """Initialse the network layers."""
        self.layer1 = Linearlayer(nb_of_inputs, nb_of_states)  # Input layer
        self.layer2 = Linearlayer(nb_of_states, nb_of_states)  # Hidden layer
        self.layer3 = Linearlayer(nb_of_states, nb_of_outputs)  # Output layer
        self.tanh = TanH()
        self.classifier = LogisticClassifier()  # Classification output
        
    def forward(self, X):
        Z1 = self.layer1.forward(X)
        Y1 = self.tanh.forward(Z1)
        Z2 = self.layer2.forward(Y1)
        Y2 = self.tanh.forward(Z2)
        Z3 = self.layer3.forward(Y2)
        Y = self.classifier.forward(Z3) 
        return Z1, Y1, Z2, Y2, Z3, Y
    
    def backward(self, X, Y, Z3, Z2, Y2, Z1, Y1, T):
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
        gZ3, gY2, gW3, gB3, gZ2, gY1, gW2, gB2, gZ1, gX, gW1, gB1 = self.backward(X, Y, Z3, Z2, Y2, Z1, Y1, T)
        #self.gW3 = gW3
        #print "gw3",gW3
        #print "gW1",gW1
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
    
    

dataN = 2000
featureN = 2
X = np.random.uniform(0, 1, (dataN, featureN))
T = np.zeros((dataN , 1))
for i in range(dataN):
    X[i][0] = randint(0,1)
    X[i][1] = randint(0,1)
    T[i] = int(X[i][0]) ^ int(X[i][1])

X_train = X
T_train = T
nb_train = dataN



# Set hyper-parameters
lmbd = 0.5  # Rmsprop lambda
learning_rate = 0.05  # Learning rate
momentum_term = 0.80  # Momentum term
eps = 1e-6  # Numerical stability term to prevent division by zero
mb_size = 100  # Size of the minibatches (number of samples)

# Create the network
nb_of_states = 5  # Number of states in the recurrent layer
FNN = FeedfowardNN(featureN,1,nb_of_states)

# Set the initial parameters
nbParameters =  sum(1 for _ in FNN.get_params_iter())  # Number of parameters in the network
maSquare = [0.0 for _ in range(nbParameters)]  # Rmsprop moving average
Vs = [0.0 for _ in range(nbParameters)]  # Velocity


ye = []
x = []
index = 0
# Iterate over some iterations
for i in range(5):
    # Iterate over all the minibatches
    for mb in range(nb_train/mb_size):
        X_mb = X_train[mb * mb_size:min((mb + 1) * mb_size,dataN-1),:]  # Input minibatch
        T_mb = T_train[mb * mb_size:min((mb + 1) * mb_size,dataN-1)]  # Target minibatch
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
        index += 1

plt.plot(x,ye)
plt.xlabel('iterater times')
plt.ylabel('lost')
#plt.ylim(0,1)
plt.show()




"""
# Set hyper-parameters
learning_rate = 0.1  # Learning rate
eps = 1e-6  # Numerical stability term to prevent division by zero
mb_size = 100  # Size of the minibatches (number of samples)
# Create the network
nb_of_states = 3  # Number of states in the recurrent layer
FNN = FeedfowardNN(featureN,1,nb_of_states)
y = []
for i in range(nb_of_states):
    y.append([])
x = []
ye = []
index = 0
cputime = time.time()
# Iterate over some iterations
for i in range(100):
    # Iterate over all the minibatches
    for mb in range(nb_train/mb_size):
        X_mb = X_train[mb:mb+mb_size,:]  # Input minibatch
        T_mb = T_train[mb:mb+mb_size]  # Target minibatch
        # Get gradients after following old velocity
        backprop_grads = FNN.getParamGrads(X_mb, T_mb)  # Get the parameter gradients    
        # Update each parameter seperately
        for pIdx, P in enumerate(FNN.get_params_iter()):
            pGradNorm = learning_rate * backprop_grads[pIdx]
            P -= pGradNorm   # Update the parameter
        
        
        x.append( index )
        ye.append(FNN.cost(FNN.getOutput(X_mb), T_mb ))
        index += 1
plt.plot(x,ye)
plt.xlabel('iterater times')
plt.ylabel('lost')
plt.show()
print time.time()-cputime,"s"
"""

# Create test samples
nb_test = 20
Xtest = []
Ttest = []
for i in range(nb_test):
    X1 = randint(0,1)
    X2 = randint(0,1)
    Xtest.append([X1,X2])
    Ttest.append(X1 ^ X2)
# Push test data through network
Y = FNN.getBinaryOutput(Xtest)
Yf = FNN.getOutput(Xtest)

# Print out all test examples
for i in range(nb_test):
    print i,Xtest[i],Ttest[i],Y[i],Yf[i]