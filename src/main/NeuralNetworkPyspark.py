## 
## Original source from https://github.com/MarvinMartin24/PySpark-Neural-Network/blob/main/NeuralNetwork.ipynb ##
##
## Modified 4/21/2022 by Trevor Holland
##

import sys
import pyspark
import numpy as np
import random

class NeuralNetworkPyspark():

    def __init__(self, n_inputs, n_hiddens_per_layer, n_outputs, activation_function='tanh'):
        self.input_layer = n_inputs # number of neurons in the input layer
        self.hidden_layer = n_hiddens_per_layer # number of neurons in the hidden layers (Custom)
        self.output_layer = n_outputs # number of neurons in the output layer
        self.act_func = activation_function #activation function

        # Initialize Weights
        self.W1 = np.random.rand(input_layer, hidden_layer) - 0.5 # Shape (784, 64)
        self.W2 = np.random.rand(hidden_layer, output_layer) - 0.5 # Shape (64, 2)
        self.B1 = np.random.rand(1, hidden_layer) - 0.5 # Shape (1, 64)
        self.B2 = np.random.rand(1, output_layer) - 0.5 # Shape (1, 2)

        self.trained = False
        self.error_trace = []
        self.Xmeans = None
        self.Xstds = None
        self.Tmeans = None
        self.Tstds = None

    '''Activation functions'''
    # General function to apply any activation function
    def activation(self, x, f):
        return f(x)

    # Sigmoid Activation function
    def sigmoid(self, X):
        return 1 / (1 + np.exp(-X))

    # Sigmoid prime function (used for backward prop)
    def sigmoid_prime(self, x):
        sig = self.sigmoid(x)
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
    def preforward(self, x, w, b):
        return np.dot(x, w) + b

    # Compute the layer propagation after activation
    # This is also equivalent to a predict function once model is trained
    def predict(self, x, W1, B1, W2, B2):
        return self.activation(self.preforward(self.activation(self.preforward(x , W1, B1)), W2, B2))

    '''Backward propogation function'''
    # Compute the derivative of the error regarding B2
    def derivativeB2(self, y_pred, y_true, y_h, f_prime):
        return (y_pred - y_true) * f_prime(y_h)

    # Compute the derivative of the error regarding W2
    def derivativeW2(self, h, dB2):
        return np.dot(h.T, dB2)

    # Compute the derivative of the error regarding B1
    def derivativeB1(self, h_h, dB2, W2, f_prime):
        return np.dot(dB2, W2.T) * f_prime(h_h)

    # Compute the derivative of the error regarding W1
    def derivativeW1(self, x, dB1):
        return np.dot(x.T, dB1)

    '''Evalutaion functions'''
    def get_metrics(self, pred, true):
        cm = [] # multilabel_confusion_matrix(true, pred)
        return (cm)

    # Cost function
    def sse(self, y_pred, y_true):
        return 0.5 * np.sum(np.power(y_pred - y_true, 2))

    '''Training function'''
    def train(self, X, T, num_iteration=50, learning_rate=0.1, verbose=True):
        # Hyperparameters
        # num_iteration = 50
        # learning_rate = 0.1
        if self.act_func=='tanh':
            act = self.tanh
            der = self.tanh_prime
        elif self.act_func == 'sig':
            act = self.sigmoid
            act = self.sigmoid_prime
        else:
            act = self.relu
            act = self.relu_prime

        input_layer = 784 # number of neurones in the input layer (equal to image size)
        hidden_layer = 64 # number of neurones in the hidden layer (Custom)
        output_layer = 2 # number of neurones in the output layer (equal to the number of possible labels)

        # Paramater Initialization
        W1 = np.random.rand(input_layer, hidden_layer) - 0.5 # Shape (784, 64)
        W2 = np.random.rand(hidden_layer, output_layer) - 0.5 # Shape (64, 2)
        B1 = np.random.rand(1, hidden_layer) - 0.5 # Shape (1, 64)
        B2 = np.random.rand(1, output_layer) - 0.5 # Shape (1, 2)

        # History over epochs
        cost_history = []
        acc_history = []

        # Epoch Loop (mini batch implementation)
        print("Start Training Loop:")

        for i in range(num_iteration):
            
            # Compute gradients, cost and accuracy over mini batch 
            
            ################## Notations ######################
            # x -> Input Image flatten of shape (1, 784)
            # y* -> One hot label of shape (1, 2)
            # h^ -> Forward prop from Input layer to hidden layer before activation (1, 64) using W1, B1 parm
            # h -> Forward prop from Input layer to hidden layer after tanh activation (1, 64)
            # y^ -> Forward prop from hidden layer to output layer before activation (1, 2) using W2, B2 parm
            # y -> Forward prop from hidden layer to output layer after sigmoid activation (1, 2)
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
                                .map(lambda x: (x[0], self.preforward(x[0], W1, B1), x[1]))\
                                .map(lambda x: (x[0], x[1], self.activation(x[1], act), x[2]))\
                                .map(lambda x: (x[0], x[1], x[2], self.preforward(x[2], W2, B2), x[3]))\
                                .map(lambda x: (x[0], x[1], x[2], x[3], self.activation(x[3], act), x[4]))\
                                .map(lambda x: (x[0], x[1], x[2], self.sse(x[4], x[5]), self.derivativeB2(x[4], x[5], x[3], der), int(np.argmax(x[4]) == np.argmax(x[5]))))\
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
                print(f"   Epoch {i+1}/{num_iteration} | Cost: {cost_history[i]} | Acc: {acc_history[i]*100} | Batchsize:{n}")

        print("Training end..")
        return self

    '''Use the model generated by train()'''
    def use(self, X):
        # Use the trained model over the Testset and get Confusion matrix per class
        metrics = test_rdd.map(lambda x: self.get_metrics(np.round(self.predict(x[0], W1, B1, W2, B2)), x[1]))\
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

