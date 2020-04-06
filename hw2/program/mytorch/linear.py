# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
import math

class Linear():
    def __init__(self, in_feature, out_feature, weight_init_fn, bias_init_fn):

        """
        Argument:
            W (np.array): (in feature, out feature)
            dW (np.array): (in feature, out feature)
            momentum_W (np.array): (in feature, out feature)

            b (np.array): (out feature)
            db (np.array): (out feature)
            momentum_B (np.array): (1, out feature)
        """

        self.W = weight_init_fn(in_feature, out_feature)
        self.b = bias_init_fn(out_feature)

        self.dW = np.zeros(self.W.shape)
        self.db = np.zeros(self.b.shape)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch size, in feature)
        Return:
            out (np.array): (batch size, out feature)
        """
        self.x = x

        # ToDo: 
        self.out = x.dot(self.W) + self.b
        return self.out
        # raise NotImplemented

    def backward(self, delta):

        """
        Argument:
            delta (np.array): (batch size, out feature)
        Return:
            out (np.array): (batch size, in feature)
        """
        
        self.db = np.dot(delta.T,np.ones(delta.shape[0])) / delta.shape[0]     # shape should be (1, out_features); divide batch_size to pass the auto_grader

        # ToDo: 
        self.dW = np.dot(self.x.T, delta) / delta.shape[0]       # divide batch_size to pass the auto_grader 
        dx = np.dot(delta, self.W.T)
        return dx
        # raise NotImplemented