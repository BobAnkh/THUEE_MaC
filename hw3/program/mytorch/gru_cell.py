import numpy as np
from activation import *

class GRU_Cell:
    """docstring for GRU_Cell"""
    def __init__(self, in_dim, hidden_dim):
        self.d = in_dim
        self.h = hidden_dim
        h = self.h
        d = self.d
        self.x_t=0

        self.Wzh = np.random.randn(h,h)
        self.Wrh = np.random.randn(h,h)
        self.Wh  = np.random.randn(h,h)

        self.Wzx = np.random.randn(h,d)
        self.Wrx = np.random.randn(h,d)
        self.Wx  = np.random.randn(h,d)

        self.dWzh = np.zeros((h,h))
        self.dWrh = np.zeros((h,h))
        self.dWh  = np.zeros((h,h))

        self.dWzx = np.zeros((h,d))
        self.dWrx = np.zeros((h,d))
        self.dWx  = np.zeros((h,d))

        self.z_act = Sigmoid()
        self.r_act = Sigmoid()
        self.h_act = Tanh()

        # Define other variables to store forward results for backward here


    def init_weights(self, Wzh, Wrh, Wh, Wzx, Wrx, Wx):
        self.Wzh = Wzh
        self.Wrh = Wrh
        self.Wh = Wh
        self.Wzx = Wzx
        self.Wrx = Wrx
        self.Wx  = Wx

    def __call__(self, x, h):
        return self.forward(x,h)

    def forward(self, x, h):
        # input:
        #   - x: shape(input dim),  observation at current time-step
        #   - h: shape(hidden dim), hidden-state at previous time-step
        #
        # output:
        #   - h_t: hidden state at current time-step

        self.x = x
        self.hidden = h

        # ToDo:
        #----------------------->
        # Define your variables based on the writeup using the corresponding
        # names below.
       # <---------------------

        # self.z = self.z_act(??? + self.Wzx.dot(x))
        # self.r = self.r_act(??? + self.Wrx.dot(x))
        # self.h_tilde = self.h_act(??? + self.Wx.dot(x))
        # h_t = (1 - self.z) * h + self.z * self.h_tilde

        assert self.x.shape == (self.d, )
        assert self.hidden.shape == (self.h, )

        assert self.r.shape == (self.h, )
        assert self.z.shape == (self.h, )
        assert self.h_tilde.shape == (self.h, )
        assert h_t.shape == (self.h, )

        return h_t
        # raise NotImplementedError

    # This must calculate the gradients wrt the parameters and return the
    # derivative wrt the inputs, xt and ht, to the cell.
    def backward(self, delta):
        # input:
        #  - delta:  shape (hidden dim), summation of derivative wrt loss from next layer at
        #            the same time-step and derivative wrt loss from same layer at
        #            next time-step
        # output:
        #  - dx: Derivative of loss wrt the input x
        #  - dh: Derivative  of loss wrt the input hidden h

        # ToDo:
        #----------------------->
        # 1) Represent each variable through a column vector 
        # 2) Compute all of the derivatives in order
        # Understand then uncomment this code, and fill in the blanks marked with '???'
        # <---------------------

        # dx = np.zeros((1, self.d))
        # dh = np.zeros((1, self.h))
        # delta = delta.reshape(-1)

        # dh += (delta * (1-self.z)).reshape(1, -1)    # (1, h)

        # dz = delta * (self.h_tilde - self.hidden)   # (h)
        # dz_inner = dz * self.z_act.derivative()       # (h)
        # self.dWzh = dz_inner.reshape(-1, 1).dot(self.hidden.reshape(1, -1))     # (h, h)
        # self.dWzx = dz_inner.reshape(-1, 1).dot(self.x.reshape(1, -1))          # (h, d)
        # dh += dz_inner.reshape(1, -1).dot(???)     # (1, h)
        # dx += dz_inner.reshape(1, -1).dot(???)     # (1, d)

        # dh_tilde = delta * self.z       # (h)
        # dh_tilde_inner = dh_tilde * self.h_act.derivative()   # (h)
        # self.dWx = dh_tilde_inner.reshape(-1, 1).dot(self.x.reshape(1, -1))     # (h, d)
        # dx += dh_tilde_inner.reshape(1, -1).dot(???)    # (1, d)

        # drh = self.Wh.T.dot(???)     # (h)
        # self.dWh = dh_tilde_inner.reshape(-1, 1).dot((self.r * self.hidden).reshape(1, -1))     # (h, h)
        # dh += (drh * self.r).reshape(1, -1)      # (1, h)

        # dr = drh * self.hidden      # (h)
        # dr_inner = dr * self.r_act.derivative()    # (h)
        # self.dWrh = dr_inner.reshape(-1, 1).dot(self.hidden.reshape(1, -1))     # (h, h)
        # self.dWrx = dr_inner.reshape(-1, 1).dot(self.x.reshape(1, -1))      # (h, d)
        # dh += dr_inner.reshape(1, -1).dot(???)     # (1, h)
        # dx += dr_inner.reshape(1, -1).dot(???)     # (1, d)

        # assert dx.shape == (1, self.d)
        # assert dh.shape == (1, self.h)

        # return dx, dh
        raise NotImplementedError
