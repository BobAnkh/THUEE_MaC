# [Acknowledgement] The source codes are adapted from CMU 11-785 Deep Learning course (http://deeplearning.cs.cmu.edu/) with prior consent of Professor Bhiksha Raj.

# DO NOT import any additional 3rd party external libraries as they are not needed (or allowed)

import numpy as np
import os
import sys

sys.path.append('mytorch')
from loss import *
from activation import *
from linear import *
from conv import *

class CNN(object):

    """
    A simple convolutional neural network

    The detailed CNN architecture is defined in function "get_cnn_model" in runner.py
    """

    def __init__(self, input_width, num_input_channels, num_channels, kernel_sizes, strides,
                 num_linear_neurons, activations, conv_weight_init_fn, bias_init_fn,
                 linear_weight_init_fn, criterion, lr):
        """
        input_width           : int    : The width of the input to the first convolutional layer
        num_input_channels    : int    : Number of channels for the input layer
        num_channels          : [int]  : List containing number of (output) channels for each conv layer
        kernel_sizes          : [int]  : List containing kernel width for each conv layer
        strides               : [int]  : List containing stride size for each conv layer
        num_linear_neurons    : int    : Number of neurons in the linear layer
        activations           : [obj]  : List of objects corresponding to the activation fn for each conv layer
        conv_weight_init_fn   : fn     : Function to init each conv layers weights
        bias_init_fn          : fn     : Function to initialize each conv layers AND the linear layers bias to 0
        linear_weight_init_fn : fn     : Function to initialize the linear layers weights
        criterion             : obj    : Object to the criterion (SoftMaxCrossEntropy) to be used
        lr                    : float  : The learning rate for the class

        You can be sure that len(activations) == len(num_channels) == len(kernel_sizes) == len(strides)
        """

        # Don't change this -->
        self.train_mode = True
        self.nlayers = len(num_channels)

        self.activations = activations
        self.criterion = criterion

        self.lr = lr
        # <---------------------

        # Don't change the name of the following class attributes,
        # the autograder will check against these attributes. But you will need to change
        # the values in order to initialize them correctly

        self.convolutional_layers = []
        self.convolutional_layers.append(Conv1D(num_input_channels,num_channels[0],kernel_sizes[0],strides[0],conv_weight_init_fn,bias_init_fn))
        
        # ToDo:
        # Hint:
        # Initializing the rest convolutional layers 
        # by calling Conv1D() with apropriate initialization parameters.
        
        # for i in range(1, self.nlayers):
        #    self.convolutional_layers.append(Conv1D(???))


        # The flatten layer will transform a two-dimensional matrix of features into a vector that can be fed into a fully connected layer.
        # The initialization of flatten layer has no parameter.
        self.flatten = Flatten()
        
        # ToDo:
        #----------------------->
        # Hint:
        # Calling Linear() with apropriate initialization parameters.        
        # As defined in  function "get_cnn_model" in runner.py,
        # The input width of data is 128
        # kernel_sizes for three convolutional layers are [5, 6, 2]
        # strides for three convolutional layers are [1, 2, 2]
        # out_channels for three convolutional layers are [56, 28, 14]
        # calculate the output width of the third convolutional layer, output_width*output_channel is the input dimension (???) of the linear layer
        # <---------------------
        
        
        # self.linear_layer = Linear(???, num_linear_neurons, linear_weight_init_fn, bias_init_fn)


    def forward(self, x):
        """
        Argument:
            x (np.array): (batch_size, num_input_channels, input_width)
        Return:
            out (np.array): (batch_size, num_linear_neurons)
        """

        # ToDo:
        #----------------------->
        # Iterate through each layer
        # Hint:
        # Calling the forward functions of convolutional layers and their acitivation funtions
        # <---------------------

        # for i in range(self.nlayers):
        #    x = ???    # convolutional layer
        #    x = ???    # acitivation funtion
            
        x = self.flatten(x)
        x = self.linear_layer(x)

        # Save output (necessary for error and loss)
        self.output = x

        return self.output

    def backward(self, labels):
        """
        Argument:
            labels (np.array): (batch_size, num_linear_neurons)
        Return:
            grad (np.array): (batch size, num_input_channels, input_width)
        """

        m, _ = labels.shape
        self.loss = self.criterion(self.output, labels).sum()
        grad = self.criterion.derivative()
        grad = self.linear_layer.backward(grad)
 
        grad = self.flatten.backward(grad)
        
        # ToDo:
        #----------------------->
        # Iterate through each layer in reverse order
        # Hint:
        # Calculating the gradation by calling the derivative of acitivation funtions and the backward functions of convolutional layers
        # <---------------------

        # for i in range(self.nlayers-1,-1,-1):
        #    grad = ??? *grad    # acitivation funtion
        #    grad = ???          # convolutional layer
        
        return grad


    def zero_grads(self):
        # Do not modify this method
        for i in range(self.nlayers):
            self.convolutional_layers[i].dW.fill(0.0)
            self.convolutional_layers[i].db.fill(0.0)

        self.linear_layer.dW.fill(0.0)
        self.linear_layer.db.fill(0.0)

    def step(self):
        # Do not modify this method
        for i in range(self.nlayers):
            self.convolutional_layers[i].W = (self.convolutional_layers[i].W -
                                              self.lr * self.convolutional_layers[i].dW)
            self.convolutional_layers[i].b = (self.convolutional_layers[i].b -
                                  self.lr * self.convolutional_layers[i].db)

        self.linear_layer.W = (self.linear_layer.W - self.lr * self.linear_layers.dW)
        self.linear_layers.b = (self.linear_layers.b -  self.lr * self.linear_layers.db)


    def __call__(self, x):
        # Do not modify this method
        return self.forward(x)

    def train(self):
        # Do not modify this method
        self.train_mode = True

    def eval(self):
        # Do not modify this method
        self.train_mode = False