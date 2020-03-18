# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
import os

# The following Criterion class will be used again as the basis for a number
# of loss functions (which are in the form of classes so that they can be
# exchanged easily (it's how PyTorch and other ML libraries do it))

class Criterion(object):
    """
    Interface for loss functions.
    """

    # Nothing needs done to this class, it's used by the following Criterion classes

    def __init__(self):
        self.logits = None
        self.labels = None
        self.loss = None

    def __call__(self, x, y):
        return self.forward(x, y)

    def forward(self, x, y):
        raise NotImplemented

    def derivative(self):
        raise NotImplemented

class SoftmaxCrossEntropy(Criterion):
    """
    Softmax loss
    """

    def __init__(self):
        super(SoftmaxCrossEntropy, self).__init__()
        self.sm = None

    def forward(self, x, y):
        """
        Argument:
            x (np.array): (batch size, 10)
            y (np.array): (batch size, 10)
        Return:
            out (np.array): (batch size, )
        """
        self.logits = x
        self.labels = y

        # LogSumExp trick: please refer to https://www.xarg.org/2016/06/the-log-sum-exp-trick-in-machine-learning/
        
        maxx = np.max(x, axis = 1)
        self.sm = maxx + np.log(np.sum(np.exp(x - maxx[:, np.newaxis]), axis=1))
        # log softmax sum
        
        # VERIFIED:
        # Hint: use self.logits, self.labels, self.sm, and np.sum(???, axis = 1)
        # return ???
        for i in range(self.sm.shape[0]):
            for j in range(self.logits.shape[1]):
                self.logits[i][j]-=self.sm[i]
        # return - np.sum(np.dot(self.labels, (self.logits - self.sm[:, np.newaxis]).T), axis=1) / self.logits.shape[0]
        return - np.diagonal(np.dot(self.labels, self.logits.T))

    def derivative(self):
        """
        Return:
            out (np.array): (batch size, 10)
        """
        # VERIFIED:
        # Hint: fill in self.logits and self.labels in the following sentence
        #return (np.exp(???) / np.exp(self.sm)[:, np.newaxis]) - ???
        return (np.exp(self.logits) / np.exp(self.sm)[:, np.newaxis]) - self.labels