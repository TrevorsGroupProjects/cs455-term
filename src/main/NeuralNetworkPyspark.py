## 
## Original source from https://github.com/MarvinMartin24/PySpark-Neural-Network/blob/main/NeuralNetwork.ipynb 
## and from https://nbviewer.org/url/www.cs.colostate.edu/~cs445/notebooks/07.2%20Optimizers,%20Data%20Partitioning,%20Finding%20Good%20Parameters.ipynb
## 
## Modified April 2022 by Trevor Holland
##

import sys
import pyspark
import numpy as np
import random

class NeuralNetworkPyspark():

    '''Initialization functions'''
    def __init__(self, n_inputs, n_outputs, n_hiddens_per_layer=[3, 10, 10], activation_function='tanh'):
        self.input_layer = n_inputs # number of neurons in the input layer
        self.output_layer = n_outputs # number of neurons in the output layer
        self.act_func = activation_function # activation function

        # Set self.n_hiddens_per_layer to [] if argument is 0, [], or [0]
        if n_hiddens_per_layer == 0 or n_hiddens_per_layer == [] or n_hiddens_per_layer == [0]:
            self.n_hiddens_per_layer = []
        else:
            self.n_hiddens_per_layer = n_hiddens_per_layer

        # Initialize weights, by first building list of all weight matrix shapes.
        n_in = n_inputs
        shapes = []
        for nh in self.n_hiddens_per_layer:
            shapes.append((n_in + 1, nh))
            n_in = nh
        shapes.append((n_in + 1, n_outputs))

        # self.all_weights:  vector of all weights
        # self.Ws: list of weight matrices by layer
        self.all_weights, self.Ws = self.make_weights_and_views(shapes)

        # Define arrays to hold gradient values.
        # One array for each W array with same shape.
        self.all_gradients, self.dE_dWs = self.make_weights_and_views(shapes)

        self.trained = False
        self.error_trace = []
        self.Xmeans = None
        self.Xstds = None
        self.Tmeans = None
        self.Tstds = None

    def printWeights(self):
        for W in self.Ws:
            print(W)
            print(W.shape)

    def make_weights_and_views(self, shapes):
        # vector of all weights built by horizontally stacking flatenned matrices
        # for each layer initialized with uniformly-distributed values.
        all_weights = np.hstack([np.random.uniform(size=shape).flat / np.sqrt(shape[0])
                                 for shape in shapes])
        # Build list of views by reshaping corresponding elements from vector of all weights
        # into correct shape for each layer.
        views = []
        start = 0
        for shape in shapes:
            size =shape[0] * shape[1]
            views.append(all_weights[start:start + size].reshape(shape))
            start += size
        return all_weights, views

    def collectMeansAndStandards(self, rdd, verbose=False):
        self.Xmeans = rdd.map(lambda x: x[0]).mean()
        self.Xstds = rdd.map(lambda x: x[0]).stdev()
        self.Xstds[self.Xstds == 0] = 1  # So we don't divide by zero when standardizing
        self.Tmeans = rdd.map(lambda x: x[1]).mean()
        self.Tstds = rdd.map(lambda x: x[1]).stdev()
        self.Tstds[self.Tstds == 0] = 1  # So we don't divide by zero when standardizing
        if verbose:
            print(self.Xmeans)
            print(self.Tmeans)
            print(self.Xstds)
            print(self.Tstds)
        return self

    def standardizeX(self, X):
        return (X - self.Xmeans) / self.Xstds

    def standardizeT(self, T):
        return (T - self.Tmeans) / self.Tstds

    '''Activation functions'''
    # General function to apply any activation function
    def activation(self, x, f):
        return f(x)

    def sigmoid(self, X):
        return 1 / (1 + np.exp(-X))

    def sigmoid_prime(self, X):
        sig = self.sigmoid(X)
        return sig * (1 - sig)

    def tanh(self, X):
        return np.tanh(X)

    def tanh_prime(self, X):
        y = self.tanh(X)
        return 1 - y*y
    
    def relu(self, s):
        s[s < 0] = 0
        return s

    def relu_prime(self, s):
        return (s > 0).astype(int)
    
    '''Forward Pass functions'''
    # Compute the layer propagation without activation
    def forward_pass(self, forward):
        if self.act_func=='tanh':
            act = self.tanh
        elif self.act_func == 'sig':
            act = self.sigmoid
        else:
            act = self.relu
        i = 0
        for W in self.Ws:
            forward = forward.map(
                lambda x, i=i, W=W: (
                    *x[:i+1], 
                    x[i] @ W[1:, :] + W[0:1, :], 
                    *x[-1:]
                )
            )\
            .map(
                lambda x, i=i: (
                    *x[:i+2], 
                    self.activation(x[i+1], act), 
                    *x[-1:]
                )
            )
            i += 2
        return forward

    '''Backward propogation functions'''
    def backward_pass(self, X):
        '''Assumes forward_pass just called with layer outputs in the rdd'''
        if self.act_func=='tanh':
            der = self.tanh_prime
        elif self.act_func == 'sig':
            der = self.sigmoid_prime
        else:
            der = self.relu_prime
        n_layers = 2 * (len(self.n_hiddens_per_layer))
        # Compute the error from the output layer
        # map the cost to the output -1 position,
        # the derivative of the output bias to the output position,
        # and map the mean square error to the end position
        backward = X.map(
            lambda x: (
                *x[:-3], 
                self.sse(x[-2], x[-1]), 
                self.derivativeBias2(x[-2], x[-1], x[-3], der), 
                x[-2] - x[-1]
            )
        )
        # Step backwards through the layers to compute the gradient derivatives
        for layeri in range(n_layers, 1, -2):
            backward = backward.map(
                lambda x, layeri=layeri: (*x[:layeri], *x[layeri+1:-1], self.derivativeWeights(x[layeri], x[-2]) ,*x[-1:])
            )
            backward = backward.map(
                lambda x, layeri=layeri: (*x[:layeri-1],*x[layeri:-1], self.derivativeBias1(x[layeri], x[-3], self.Ws[layeri//2][1:, :], der) ,*x[-1:])
            )
        # Compute the final derivative
        backward = backward.map(lambda x: (*x[1:-1], self.derivativeWeights(x[0], x[-2]) ,*x[-1:], 1))
        return backward

    # Compute the derivative of the error regarding biases
    def derivativeBias1(self, h_h, dB, W, f_prime):
        return np.dot(dB, W.T) * f_prime(h_h)

    # Compute the derivative of the error regarding the final bias
    def derivativeBias2(self, y_pred, y_true, y_h, f_prime):
        return (y_pred - y_true) * f_prime(y_h)

    # Compute the derivative of the error regarding Weights
    def derivativeWeights(self, layer, dB):
        return np.dot(layer.T, dB)

    '''Evalutaion functions'''
    # Cost (sum of squared errors) function
    def sse(self, y_pred, y_true):
        return 0.5 * np.sum(np.power(y_pred - y_true, 2))

    # Mean squared error
    def mse(self, Y, T):
        mean_sq_error = np.mean((T - Y) ** 2)
        return mean_sq_error

    '''Training functions'''
    def train(self, train_rdd, num_epochs=50, learning_rate=0.1, verbose=True):
        # Setup standardization parameters
        if self.Xmeans is None:
            self.collectMeansAndStandards(train_rdd)
        print("Standardizing X and T...")
        # Standardize X and T
        train_rdd = train_rdd.map(lambda x: (self.standardizeX(x[0]), self.standardizeT(x[1]) ))
        print("X and T have been standardized.")
        # History over epochs
        self.cost_history = []
        self.acc_history = []

        # Epoch Loop (mini batch implementation)
        print("Start Training Loop:")

        for epoch in range(num_epochs):
            
            # Compute gradients, cost, and error over mini batch 
            # print("Running a forward pass")
            forward_rdd = self.forward_pass(train_rdd)
            # print("Running a backward pass")
            backward_rdd = self.backward_pass(forward_rdd)
            # print("Collecting results")
            def tupleAdder(tuple1, tuple2):
                return tuple(map(lambda x, y: x + y, tuple1, tuple2))
            gradientCostAcc = backward_rdd.reduce(lambda x,y: tupleAdder(x,y))

            # Cost and Error of the mini batch
            n = gradientCostAcc[-1] # number of samples in the mini batch
            cost = gradientCostAcc[0]/n # Cost over the mini batch
            acc = gradientCostAcc[-2]/n # accuracy over the mini batch
            
            # print(len(gradientCostAcc))

            # Add to history
            self.cost_history.append(cost)
            self.acc_history.append(acc)

            # Extract gradients
            r_bias_and_weight =  list(reversed(gradientCostAcc[1:-2]))
            i = 0
            for dW in self.dE_dWs:
                dW[1:, :] = r_bias_and_weight[i]/n
                dW[0:1, :] = r_bias_and_weight[i+1]/n
                i += 2
                    
            # Update parameters with learning rate and gradients using Gradient Descent
            self.all_weights -= learning_rate * self.all_gradients

            # Display performance
            if verbose:
                print(f"   Epoch {epoch+1}/{num_epochs} | Cost: {self.cost_history[epoch]} | Error: {self.acc_history[epoch]} | Batchsize:{n}")

        print("Training end..")
        self.trained = True
        return self

    '''Use the model generated by train()'''
    def use(self, use_rdd):
        # Use the trained model over the use_rdd
        if self.trained:
            use_rdd = use_rdd.map(lambda x: (self.standardizeX(x[0]), self.standardizeT(x[1]) ))
        result = self.forward_pass(use_rdd)
        result = result.map(lambda x: (x[-2]))
        if self.trained:
            def unstandardize(Y):
                return Y * self.Tstds + self.Tmeans
            result = result.map(lambda x: (unstandardize(x)))
        result = result.collect()
        return result
        
