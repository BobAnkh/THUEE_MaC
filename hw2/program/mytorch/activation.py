# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
import os


class Activation(object):
    """
    Interface for activation functions (non-linearities).

    In all implementations, the state attribute must contain the result,
    i.e. the output of forward (it will be tested).
    """

    # No additional work is needed for this class, as it acts like an
    # abstract base class for the others

    # Note that these activation functions are scalar operations. I.e, they
    # shouldn't change the shape of the input.

    def __init__(self):
        self.state = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        raise NotImplemented

    def derivative(self):
        raise NotImplemented


class Identity(Activation):
    """
    Identity function (already implemented).
    """

    # This class is a gimme as it is already implemented for you as an example

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        self.state = x
        return x

    def derivative(self):
        return 1.0


class Sigmoid(Activation):
    """
    Sigmoid non-linearity
    """

    # Remember do not change the function signatures as those are needed
    # to stay the same for AutoLab.

    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, x):
        # DONE:
        # Might we need to store something before returning?
        # self.state = ???
        # Hint: You can use np.exp() function
        # return self.state
        self.state = x
        self.state[x>=0] = 1.0 / (1.0 + np.exp(- self.state[x>=0]))
        self.state[x<0] = np.exp(self.state[x<0]) / ( np.exp(self.state[x<0]) + 1.0)
        return self.state

    def derivative(self):
        # DONE:
        # Maybe something we need later in here...
        # return ???
        # Maybe something we need later in here...
        return self.state * (1.0 - self.state)


class Tanh(Activation):
    """
    Tanh non-linearity
    """
    def __init__(self):
        super(Tanh, self).__init__()

    def forward(self, x):
        # DONE:
        # self.state = ???
        # Hint: You can use np.exp() function
        # return self.state
        self.state = (1 - np.exp(-2 * x)) / (1 + np.exp(-2 * x))
        return self.state

    def derivative(self):
        # DONE:
        # return ???
        return (1 - self.state**2)


class ReLU(Activation):
    """
    ReLU non-linearity
    """
    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, x):
        # DONE:
        # self.state = ???
        # return self.state
        self.state = np.maximum(x, 0)
        return self.state

    def derivative(self):
        # DONE
        # return ???
        tmp = self.state
        tmp[tmp > 0] = 1
        return tmp