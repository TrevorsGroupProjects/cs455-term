## 
## Original source from https://github.com/MarvinMartin24/PySpark-Neural-Network/blob/main/NeuralNetwork.ipynb 
## and from https://nbviewer.org/url/www.cs.colostate.edu/~cs445/notebooks/07.2%20Optimizers,%20Data%20Partitioning,%20Finding%20Good%20Parameters.ipynb
## 
## Modified 4/21/2022 by Trevor Holland
##

import sys
import pyspark
import numpy as np
import random

class NeuralNetworkPyspark():

    '''Initialization functions'''
    def __init__(self, n_inputs, n_outputs, n_hiddens_per_layer=[3, 10, 10], activation_function='tanh'):
        self.input_layer = n_inputs # number of neurons in the input layer
        self.hidden_layer = n_hiddens_per_layer # number of neurons in the hidden layers (Custom)
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

    # Sigmoid Activation function
    def sigmoid(self, X):
        return 1 / (1 + np.exp(-X))

    # Sigmoid prime function (used for backward prop)
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
    # Compute the layer propagation before activation
    def preforward(self, X, w, b):
        return np.dot(X, w) + b

    # Compute the layer propagation after activation
    # This is also equivalent to a predict function once model is trained
    def predict(self, x, W1, B1, W2, B2):
        return self.activation(self.preforward(self.activation(self.preforward(x , W1, B1)), W2, B2))

    '''Backward propogation function'''
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
        if self.act_func=='tanh':
            act = self.tanh
            der = self.tanh_prime
        elif self.act_func == 'sig':
            act = self.sigmoid
            der = self.sigmoid_prime
        else:
            act = self.relu
            der = self.relu_prime

        # History over epochs
        cost_history = []
        acc_history = []

        # Epoch Loop (mini batch implementation)
        print("Start Training Loop:")

        for epoch in range(num_epochs):
            
            # Compute gradients, cost and accuracy over mini batch 
            
            ################## Notations ######################
            # x -> Input Image flatten of shape (1, 784)

            # y* -> One hot label of shape # need to eliminate this
            
            # h^ -> Forward prop from Input layer to hidden layer before activation using W1, B1 parm
            # h -> Forward prop from Input layer to hidden layer after tanh activation 
            # y^ -> Forward prop from hidden layer to output layer before activation using W2, B2 parm
            # y -> Forward prop from hidden layer to output layer after sigmoid activation 
            # E -> Error between y and y* using SSE
            # Acc -> 1 is right prediction 0 otherwise
            # DE/D? -> Partial derivative of the Error regarding parmaters (B2, W2, B1, W1)
            
            
            ################# Forward Prop ######################
            # map batch ([x], [y*]) to ([x], [h^],[y*])
            # map batch ([x], [h^],[y*]) to ([x], [h^], [h], [y*])
            # map batch ([x], [h^], [h], [y*]) to ([x], [h^], [h], [y^], [y*])
            # map batch ([x], [h^], [h], [y^], [y*]) to ([x], [h^], [h], [y^], [y], [y*])
            ################# Backward Prop #####################
            # map batch ([x], [h^], [h], [y^], [y], [y*]) to ([x], [h^], [h], [E], [DE/DB2], [Acc])
            # map batch ([x], [h^], [h], [E], [DE/DB2], [Acc]) to ([x], [h^], [E], [DE/DB2], [DE/DW2], [Acc])
            # map batch ([x], [h^], [E], [DE/DB2], [DE/DW2], [Acc]) to ([x], [E], [DE/DB2], [DE/DW2], [DE/DB1], [Acc])
            # map batch ([x], [E], [DE/DB2], [DE/DW2], [DE/DB1], [Acc]) to ([E], [DE/DB2], [DE/DW2], [DE/DB1], [DE/DW1],[Acc])
            ############### Reduce over the mini batch #########


            gradientCostAcc = train_rdd\
                                .sample(False,0.7)\
                                .map(lambda x: (x[0], self.preforward(x[0], self.Ws[0][1:, :], self.Ws[0][0:1, :] ), x[1]))\
                                .map(lambda x: (x[0], x[1], self.activation(x[1], act), x[2]))\
                                .map(lambda x: (x[0], x[1], x[2], self.preforward(x[2], self.Ws[1][1:, :], self.Ws[1][0:1, :]), x[3]))\
                                .map(lambda x: (x[0], x[1], x[2], x[3], self.activation(x[3], act), x[4]))\
                                .map(lambda x: (x[0], x[1], x[2], self.mse(x[4], x[5]), self.derivativeB2(x[4], x[5], x[3], der), int(np.argmax(x[4]) == np.argmax(x[5]))))\
                                .map(lambda x: (x[0], x[1], x[3], x[4], self.derivativeW2(x[2], x[4]) ,x[5]))\
                                .map(lambda x: (x[0], x[2], x[3], x[4], self.derivativeB1(x[1],  x[3], W2, der) ,x[5]))\
                                .map(lambda x: (x[1], x[2], x[3], x[4], self.derivativeW1(x[0], x[4]) ,x[5], 1)) \
                                .reduce(lambda x, y: (x[0] + y[0], x[1] + y[1], x[2] + y[2], x[3] + y[3], x[4] + y[4], x[5] + y[5], x[6] + y[6]))

            # Cost and Accuarcy of the mini batch
            n = gradientCostAcc[-1] # number of images in the mini batch
            cost = gradientCostAcc[0]/n # Cost over the mini batch
            acc = gradientCostAcc[5]/n # Accuarcy over the mini batch
            
            # Add to history
            cost_history.append(cost)
            acc_history.append(acc)
            
            # Extract gradiends
            DB2 = gradientCostAcc[1]/n
            DW2 = gradientCostAcc[2]/n
            DB1 = gradientCostAcc[3]/n
            DW1 = gradientCostAcc[4]/n
                    
            # Update parameter with new learning rate and gradients using Gradient Descent
            B2 -= learning_rate * DB2
            W2 -= learning_rate * DW2
            B1 -= learning_rate * DB1
            W1 -= learning_rate * DW1

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

