# DO NOT import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
import os
import sys

sys.path.append('mytorch')
from loss import *
from activation import *
from linear import *
from conv import *


class CNN_SimpleScanningMLP():
    def __init__(self):

        self.conv1 = Conv1D(24, 8, 8, 4)
        self.conv2 = Conv1D(8, 16, 1, 1)
        self.conv3 = Conv1D(16, 4, 1, 1)
        self.layers = [
            self.conv1,
            ReLU(),
            self.conv2,
            ReLU(),
            self.conv3,
            Flatten()
        ]

    def __call__(self, x):
        # Do not modify this method
        return self.forward(x)

    def init_weights(self, weights):

        # ToDo:
        #----------------------->
        # Hint:
        # Load the weights for your CNN from the MLP Weights given
        # weights[0], weights[1], weights[2] contain the weights for the three layers of the MLP
        # Load them appropriately into the CNN
        # <---------------------

        self.conv1.W = weights[0].reshape(8, 24, 8).transpose(2, 1, 0)
        self.conv2.W = weights[1].T.reshape(16, 8, 1)
        self.conv3.W = weights[2].T.reshape(???, ???, ???)

    def forward(self, x):
        """
        Do not modify this method

        Argument:
            x (np.array): (batch size, in channel, in width)
        Return:
            out (np.array): (batch size, out channel , out width)
        """

        out = x
        for layer in self.layers:
            out = layer(out)
        return out

    def backward(self, delta):
        """
        Do not modify this method

        Argument:
            delta (np.array): (batch size, out channel, out width)
        Return:
            dx (np.array): (batch size, in channel, in width)
        """

        for layer in self.layers[::-1]:
            delta = layer.backward(delta)
        return delta


class CNN_DistributedScanningMLP():
    def __init__(self):
        
        self.conv1 = Conv1D(24, 2, 2, 2)
        self.conv2 = Conv1D(2, 8, 2, 2)
        self.conv3 = Conv1D(8, 4, 2, 1)
     
        self.layers = [
            self.conv1,
            ReLU(),
            self.conv2,
            ReLU(),
            self.conv3,
            Flatten()
        ]
    def __call__(self, x):
        # Do not modify this method
        return self.forward(x)

    def init_weights(self, weights):
        
        # ToDo:
        #----------------------->
        # Hint:
        # Load the weights for your CNN from the MLP Weights given
        # weights[0], weights[1], weights[2] contain the weights for the three layers of the MLP
        # Load them appropriately into the CNN
        # <---------------------

        self.conv1.W = weights[0].reshape(8, 24, 8)[:2, :, :2].transpose(2, 1, 0)
        self.conv2.W = weights[1][:4, :8].reshape(2, 2, 8).transpose(2, 1, 0)
        self.conv3.W = weights[2].reshape(???, ???, ???).transpose(2, 1, 0)

    def forward(self, x):
        """
        Do not modify this method

        Argument:
            x (np.array): (batch size, in channel, in width)
        Return:
            out (np.array): (batch size, out channel , out width)
        """

        out = x
        for layer in self.layers:
            out = layer(out)
        return out

    def backward(self, delta):
        """
        Do not modify this method

        Argument:
            delta (np.array): (batch size, out channel, out width)
        Return:
            dx (np.array): (batch size, in channel, in width)
        """

        for layer in self.layers[::-1]:
            delta = layer.backward(delta)
        return delta
