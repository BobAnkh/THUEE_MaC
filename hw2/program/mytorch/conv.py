# [Acknowledgement] The source codes are adapted from CMU 11-785 Deep Learning course (http://deeplearning.cs.cmu.edu/) with prior consent of Professor Bhiksha Raj.

# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np


class Conv1D():
    def __init__(self, in_channel, out_channel, kernel_size, stride,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride

        if weight_init_fn is None:
            self.W = np.random.normal(0, 1.0, (out_channel, in_channel, kernel_size))
        else:
            self.W = weight_init_fn(out_channel, in_channel, kernel_size)
        
        if bias_init_fn is None:
            self.b = np.zeros(out_channel)
        else:
            self.b = bias_init_fn(out_channel)

        self.dW = np.zeros(self.W.shape)
        self.db = np.zeros(self.b.shape)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch_size, in_channel, input_size)
        Return:
            out (np.array): (batch_size, out_channel, output_size)
        """
        self.batch, __, self.width = x.shape
        assert __ == self.in_channel, 'Expected the inputs to have {} channels'.format(self.in_channel)

        out_width = (self.width - self.kernel_size) // self.stride + 1
        self.out_width = out_width
        self.x = x
        result = np.zeros(shape=[self.batch, self.out_channel, out_width])      # NOTE: do NOT use np.float32
        
        # ToDo:
        #----------------------->
        # Hint:
        # Calculate the output of convolutional layer by using self.W[out_c], self.b[out_c], and x with appropriate matrix slicing
        # Use the .sum() method of a array to get the sum of array elements 
        # <---------------------
        
        # for n in range(self.batch):
        #    for out_c in range(self.out_channel):
        #        for out_pos in range(out_width):
        #            result[n, out_c, out_pos] = ???

        #return result

        raise NotImplemented



    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch_size, out_channel, output_size)
        Return:
            dx (np.array): (batch_size, in_channel, input_size)
        """
        dx = np.zeros(shape=self.x.shape)   # NOTE: do NOT use np.float32, use np.float64 instead
 
        # ToDo:
        #----------------------->
        # Hint:
        # Calculate the the gradients by filling self.kernel_size and self.stride in appropriate places (???)
        # <---------------------
        # for n in range(self.batch):
        #    for out_c in range(self.out_channel):
        #        for out_pos in range(self.out_width):
        #            self.db[out_c] += delta[n, out_c, out_pos] * 1.0
        #            self.dW[out_c] += delta[n, out_c, out_pos] * self.x[n, :, out_pos*???:out_pos*??? + ???]
        #            dx[n, :, out_pos*???:out_pos*??? + ???] += delta[n, out_c, out_pos] * self.W[out_c]

        # return dx

        raise NotImplemented



class Flatten():
    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch_size, in_channel, in_width)
        Return:
            out (np.array): (batch_size, in_channel * in width)
        """
        self.b, self.c, self.w = x.shape

        self.x = x
        return x.reshape(x.shape[0], -1)    # (N, C, W) -> (N, C*W)
        
    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch size, in channel * in width)
        Return:
            dx (np.array): (batch size, in channel, in width)
        """
        return np.reshape(delta, self.x.shape)