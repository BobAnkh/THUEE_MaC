"""
Follow the instructions provided in the writeup to completely
implement the class specifications for a basic MLP, optimizer, .
You will be able to test each section individually by submitting
to autolab after implementing what is required for that section
-- do not worry if some methods required are not implemented yet.

Notes:

The __call__ method is a special reserved method in
python that defines the behaviour of an object when it is
used as a function. For example, take the Linear activation
function whose implementation has been provided.

# >>> activation = Identity()
# >>> activation(3)
# 3
# >>> activation.forward(3)
# 3
"""

# DO NOT import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
import os
import sys

sys.path.append('mytorch')
from loss import *
from activation import *
from batchnorm import *
from linear import *


class MLP(object):

    """
    A simple multilayer perceptron
    """

    def __init__(self, input_size, output_size, hiddens, activations, weight_init_fn,
                 bias_init_fn, criterion, lr, momentum=0.0, num_bn_layers=0):

        # Don't change this -->
        self.train_mode = True
        self.num_bn_layers = num_bn_layers
        self.bn = num_bn_layers > 0
        self.nlayers = len(hiddens) + 1
        self.input_size = input_size
        self.output_size = output_size
        self.activations = activations
        self.criterion = criterion
        self.lr = lr
        self.momentum = momentum
        # <---------------------

        # Don't change the name of the following class attributes,
        # the autograder will check against these attributes. But you will need to change
        # the values in order to initialize them correctly

        t_size = [input_size]
        for i in hiddens:
            t_size.append(i)
        t_size.append(output_size) # [input, hiddens, output]

        # ToDo:
        # Initialize and add all your linear layers into the list 'self.linear_layers'
        # (HINT: self.linear_layers = [ Linear(???, ???, ???) for ?? in ? ])
        # self.linear_layers = ???

        # If batch norm, add batch norm layers into the list 'self.bn_layers'
        if self.bn:
            self.bn_layers = [BatchNorm(t_size[i+1]) for i in range(self.num_bn_layers)]

        if self.momentum != 0:
            t_size = [input_size]
            for i in hiddens:
                t_size.append(i)
            t_size.append(output_size)
            self.vW = [np.zeros((t_size[i], t_size[i + 1])) for i in range(len(t_size) - 1)]
            t_size = t_size[1:]
            self.vb = [np.zeros(i) for i in t_size]
            if self.bn:
                self.vgamma = [np.zeros(t_size[i]) for i in range(self.num_bn_layers)]
                self.vbeta = [np.zeros(t_size[i]) for i in range(self.num_bn_layers)]

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch size, input_size)
        Return:
            out (np.array): (batch size, output_size)
        """
        # Complete the forward pass through your entire MLP.
        self.hiddens = []
        self.input = x
        self.out = x
        for i in range(self.nlayers):
            self.hiddens.append(self.out)

            # ToDo: 
            # Hint: use self.linear_layers[i] and self.hiddens[i]
            # self.out = ???
            if self.bn and i < self.num_bn_layers:
                self.out = self.bn_layers[i](self.out, eval=not self.train_mode) # BN layers have different behavior according to train mode
            # Hint: use self.activations[i] and previous self.out
            # self.out = ???
        #return self.out
        raise NotImplemented

    def zero_grads(self):
        # Use numpyArray.fill(0.0) to zero out your backpropped derivatives in each
        # of your linear and batchnorm layers.
        for i in range(self.nlayers):
            self.linear_layers[i].dW = np.zeros(self.linear_layers[i].dW.shape)
            # ToDo: 
            # self.linear_layers[i].db = ???
            if self.bn and i < self.num_bn_layers:
                self.bn_layers[i].dgamma = np.zeros(self.bn_layers[i].dgamma.shape)
                # ToDo: 
                # self.bn_layers[i].dbeta = ???
        raise NotImplemented

    def step(self):
        # Apply a step to the weights and biases of the linear layers.
        # Apply a step to the weights of the batchnorm layers.
        # (You will add momentum later in the assignment to the linear layers only
        # , not the batchnorm layers)

        for i in range(len(self.linear_layers)):
            # Update weights and biases here
            pass
        # Do the same for batchnorm layers

        if self.momentum != 0:
            for i in range(self.nlayers):
                self.vW[i] = self.vW[i] * self.momentum + self.linear_layers[i].dW
                # ToDo: 
                # self.vb[i] = ???
                if self.bn and i < self.num_bn_layers:
                    self.vgamma[i] = self.vgamma[i] * self.momentum + self.bn_layers[i].dgamma
                    # ToDo:
                    # self.vbeta[i] = ???

                self.linear_layers[i].W -= self.vW[i] * self.lr
                # ToDo:
                # self.linear_layers[i].b -= ???
                if self.bn and i < self.num_bn_layers:
                    self.bn_layers[i].gamma -= self.vgamma[i] * self.lr
                    # ToDo:
                    # self.bn_layers[i].beta -= ???
        else:
            for i in range(self.nlayers):
                self.linear_layers[i].W -= self.linear_layers[i].dW * self.lr
                self.linear_layers[i].b -= self.linear_layers[i].db * self.lr
                if self.bn and i < self.num_bn_layers:
                    self.bn_layers[i].gamma -= self.bn_layers[i].dgamma * self.lr
                    self.bn_layers[i].beta -= self.bn_layers[i].dbeta * self.lr

        raise NotImplemented

    def backward(self, labels):
        # Backpropagate through the activation functions, batch norm and
        # linear layers.
        # Be aware of which return derivatives and which are pure backward passes
        # i.e. take in a loss w.r.t it's output.
        loss = self.criterion(self.out, labels) # call self.criterion.forward() explicitly to pass the auto_grader
        deriv = self.criterion.derivative()

        for i in range(self.nlayers)[::-1]:  # Backpropogation
            act_deriv = deriv * self.activations[i].derivative()
            if self.bn and i < self.num_bn_layers:
                bn_deriv = self.bn_layers[i].backward(act_deriv)
            else:
                bn_deriv = act_deriv

            # ToDo:
            # Hint: use self.linear_layers[i].backward()
            # deriv = ???

        raise NotImplemented

    def error(self, labels):
        return (np.argmax(self.output, axis = 1) != np.argmax(labels, axis = 1)).sum()

    def total_loss(self, labels):
        return self.criterion(self.output, labels).sum()

    def __call__(self, x):
        return self.forward(x)

    def train(self):
        self.train_mode = True

    def eval(self):
        self.train_mode = False

def get_training_stats(mlp, dset, nepochs, batch_size):
    train, val, test = dset
    trainx, trainy = train
    valx, valy = val
    testx, testy = test

    idxs = np.arange(len(trainx))

    training_losses = []
    training_errors = []
    validation_losses = []
    validation_errors = []

    # Setup ...

    for e in range(nepochs):

        training_losses.append(0)
        training_errors.append(0)
        validation_losses.append(0)
        validation_errors.append(0)

        seed = np.random.randint(0, 1000)
        np.random.seed(seed)
        np.random.shuffle(trainx)
        np.random.seed(seed)
        np.random.shuffle(trainy)
        # Per epoch setup ...

        for b in range(0, len(trainx), batch_size):

            # Train ...
            mlp.train()
            end = min(b+batch_size, len(trainx) -  1)
            # ToDo:
            # Hint: call mlp(), the parameter is the current batch of data trainx[b:end]
            # out = ???
            loss = mlp.criterion(out, trainy[b:end])
            loss = np.sum(loss)
            
            mlp.zero_grads()
            # ToDo:
            # Hint: call mlp.backward(), the parameter is the current batch of ground truch trainy[b:end]
            # ???
            mlp.step()

            result = np.argmax(out, axis=1)
            for i in range(len(result)):
                if result[i] != np.argmax(trainy[b + i]):
                    training_errors[e] += 1

            result = np.eye(trainy.shape[1])[result]
            training_losses[e] += loss

        for b in range(0, len(valx), batch_size):

            # Val ...
            mlp.eval()
            end = min(b+batch_size, len(valx) -  1)
            out = mlp(valx[b:end])
            loss = mlp.criterion(out, valy[b:end])
            loss = np.sum(loss)
            
            result = np.argmax(out, axis=1)
            for i in range(len(result)):
                if result[i] != np.argmax(valy[b + i]):
                    validation_errors[e] += 1

            result = np.eye(valy.shape[1])[result]
            validation_losses[e] += loss
        
        training_losses[e] /= len(trainx)
        training_errors[e] /= len(trainx)
        validation_losses[e] /= len(valx)
        validation_errors[e] /= len(valx)
        # Accumulate data...

        print("Epoch %03d  " %e, "Training: Loss %1.4f  Error  %1.4f;  " % (training_losses[e],training_errors[e]),"Validation: Loss %1.4f  Error  %1.4f;" % (validation_losses[e],validation_errors[e]))

    # Cleanup ...
    results = []
    for b in range(0, len(testx), batch_size):

        mlp.eval()
        end = min(b+batch_size, len(testx) -  1)
        out = mlp(testx[b:end])
        
        result = np.argmax(out, axis=1)
        result = np.eye(valy.shape[1])[result]

    # return training_losses, training_errors, validation_losses, validation_errors
    return training_losses, training_errors, validation_losses, validation_errors