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
        self.hidden_layer = n_hiddens_per_layer # list of number of neurons in each of the hidden layers
        self.output_layer = n_outputs # number of neurons in the output layer
        self.act_func = activation_function #activation function

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
    def forward_pass(self, X):
        if self.act_func=='tanh':
            act = self.tanh
        elif self.act_func == 'sig':
            act = self.sigmoid
        else:
            act = self.relu
        i = 0
        forward = X
        for W in self.Ws[:-1]:
            forward = forward.map(
                lambda x: (
                    x[:i], 
                    self.preforward(x[i], W[1:, :], W[0:1, :]), 
                    x[i+1]
                )
            )\
            .map(
                lambda x: (
                    x[:i+1], 
                    self.activation(x[i+1], act), 
                    x[i+2]
                )
            )
            i += 1
        forward = forward.map(
            lambda x: (
                x[:i], 
                self.preforward(x[i], W[1:, :], W[0:1, :] ), 
                x[i+1]
            )
        )
        return forward

    # Compute the layer propagation without activation
    def preforward(self, X, w, b):
        return np.dot(X, w) + b
    
    '''Delete predict?'''
    # # Compute the layer propagation after activation
    # # This is also equivalent to a predict function once model is trained
    # def predict(self, x, W1, B1, W2, B2):
    #     return self.activation(self.preforward(self.activation(self.preforward(x , W1, B1)), W2, B2))

    '''Backward propogation functions'''
    def backward_pass(self, X, T):
        '''Assumes forward_pass just called with layer outputs in the rdd'''
        # error = T - self.Ys[-1]
        # n_samples = X.shape[0]
        # n_outputs = T.shape[1]
        # delta = - error / (n_samples * n_outputs)
        # n_layers = len(self.n_hiddens_per_layer) + 1
        # # Step backwards through the layers to back-propagate the error (delta)
        # for layeri in range(n_layers - 1, -1, -1):
        #     # gradient of all but bias weights
        #     self.dE_dWs[layeri][1:, :] = self.Ys[layeri].T @ delta
        #     # gradient of just the bias weights
        #     self.dE_dWs[layeri][0:1, :] = np.sum(delta, 0)
        #     # Back-propagate this layer's delta to previous layer
        #     if self.activation_function == 'relu':
        #         delta = delta @ self.Ws[layeri][1:, :].T * self.grad_relu(self.Ys[layeri])
        #     else:
        #         delta = delta @ self.Ws[layeri][1:, :].T * (1 - self.Ys[layeri] ** 2)

        if self.act_func=='tanh':
            der = self.tanh_prime
        elif self.act_func == 'sig':
            der = self.sigmoid_prime
        else:
            der = self.relu_prime
        n_layers = 2 * (len(self.n_hiddens_per_layer) + 1)
        i = 2 * (len(self.n_hiddens_per_layer)  + 1)
        # Compute the error from the output layer
        # map the cost to the output -1 position,
        # the derivative of the output bias to the output position,
        # and map the mean square error to the output +1 position
        backward = X.map(
            lambda x: (
                x[:i-1], 
                self.sse(x[i], x[i+1]), 
                self.derivativeBias2(x[i], x[i+1], x[i-1], der), 
                self.mse(x[i], x[i+1])
            )
        )
        # Step backwards through the layers to back-propagate the error
        ### Update with correct delta, weights, and derivations
        ### Also update with correct x[] values
        for layeri in range(n_layers - 1, -1, -1):
            backward = backward.map(
                lambda x: (x[0], x[1], x[3], x[4], self.derivativeWeights(x[2], x[4]) ,x[5])
            )\
            .map(
                lambda x: (x[0], x[2], x[3], x[4], self.derivativeBias1(x[1],  x[3], self.Ws[layeri], der) ,x[5])
            )
            i -= 1
        backward = backward.map(lambda x: (x[1], x[2], x[3], x[4], self.derivativeWeights(x[0], x[4]) ,x[5], 1))
        return backward

    # Compute the derivative of the error regarding B1
    def derivativeBias1(self, h_h, dB, W, f_prime):
        return np.dot(dB, W.T) * f_prime(h_h)

    # Compute the derivative of the error regarding B2
    def derivativeBias2(self, y_pred, y_true, y_h, f_prime):
        return (y_pred - y_true) * f_prime(y_h)

    # Compute the derivative of the error regarding Weights
    def derivativeWeights(self, h, dB):
        return np.dot(h.T, dB)

    '''Evalutaion functions'''
    def get_metrics(self, pred, true):
        cm = [] # multilabel_confusion_matrix(true, pred)
        return (cm)

    # Cost function
    def sse(self, y_pred, y_true):
        return 0.5 * np.sum(np.power(y_pred - y_true, 2))

    # Mean squared error
    def mse(self, Y, T):
        mean_sq_error = np.mean((T - Y) ** 2)
        return mean_sq_error

    '''Training functions'''
    def train(self, train_rdd, num_epochs=50, learning_rate=0.1, verbose=True):
        # History over epochs
        cost_history = []
        acc_history = []

        # Epoch Loop (mini batch implementation)
        print("Start Training Loop:")

        for epoch in range(num_epochs):
            
            # Compute gradients, cost and accuracy over mini batch 
            forward_rdd = self.forward_pass(train_rdd.sample(False,0.7))
            backward_rdd = self.backward_pass(forward_rdd)
            
            # This needs to be generalized to the size of the network
            gradientCostAcc = backward_rdd.reduce(lambda x, y: (x[0] + y[0], x[1] + y[1], x[2] + y[2], x[3] + y[3], x[4] + y[4], x[5] + y[5], x[6] + y[6]))

            # Cost and Accuarcy of the mini batch
            n = gradientCostAcc[-1] # number of images in the mini batch
            cost = gradientCostAcc[0]/n # Cost over the mini batch
            acc = gradientCostAcc[5]/n # Accuarcy over the mini batch
            
            # Add to history
            cost_history.append(cost)
            acc_history.append(acc)
            
            # Extract gradiends
            ### self.all_gradients #???
                    
            # Update parameter with new learning rate and gradients using Gradient Descent
            ### self.all_weights #???

            # Display performances
            if verbose:
                print(f"   Epoch {epoch+1}/{num_epochs} | Cost: {cost_history[epoch]} | Acc: {acc_history[epoch]*100} | Batchsize:{n}")

        print("Training end..")
        return self

    '''Use the model generated by train()'''
    def use(self, use_rdd):
        # Use the trained model over the Testset and get Confusion matrix per class
        metrics = use_rdd.map(lambda x: self.get_metrics(np.round(self.predict(x[0], W1, B1, W2, B2)), x[1]))\
                        .reduce(lambda x, y: x + y)

        # For each class give TP, FP, FN, TN and precision, and recall, and F1 score
        for label, label_metrics in enumerate(metrics):
            
            print(f"\n---- Digit {label} ------\n")
            tn, fp, fn, tp = label_metrics.ravel()
            print("TP:", tp, "FP:", fp, "FN:", fn, "TN:", tn)

            precision = tp / (tp + fp + 0.000001)
            print(f"\nPrecision : {precision}")

            recall = tp / (tp + fn + 0.000001)
            print(f"Recall: {recall}")

            F1 = 2 * (precision * recall) / (precision + recall + 0.000001)
            print(f"F1 score: {F1}")

