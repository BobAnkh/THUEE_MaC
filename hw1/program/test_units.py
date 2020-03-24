import numpy as np
import os
import sys
sys.path.append('hw1/program/autograder/hw1_autograder/tests')
from helpers.helpers import *

import pickle

saved_data = pickle.load(open("hw1/program/autograder/hw1_autograder/data.pkl", 'rb'))
rtol = 1e-4
atol = 1e-04
TOLERANCE = 1e-4
SEED = 2019

sys.path.append('hw1/program/mytorch')
import activation
import loss
import linear
import batchnorm

sys.path.append('hw1/program/hw1')
import hw1


def test_softmax_cross_entropy_forward():
    data = saved_data[0]
    x = data[0]
    y = data[1]
    sol = data[2]

    ce = loss.SoftmaxCrossEntropy()
    closeness_test(ce(x, y), sol, "ce(x, y)")
    print('softmax cross entropy forward done!')


def test_softmax_cross_entropy_derivative():
    data = saved_data[1]
    x = data[0]
    y = data[1]
    sol = data[2]
    ce = loss.SoftmaxCrossEntropy()
    ce(x, y)
    closeness_test(ce.derivative(), sol, "ce.derivative()")


def test_relu_forward():
    data = saved_data[7]
    t0 = data[0]
    gt = data[1]
    student = activation.ReLU()
    student(t0)
    closeness_test(student.state, gt, "relu.state")
    print('relu forward done!')


def test_relu_derivative():
    data = saved_data[8]
    t0 = data[0]
    gt = data[1]
    student = activation.ReLU()
    student(t0)
    closeness_test(student.derivative(), gt, "relu.derivative()")
    print('relu derivative done!')


def test_sigmoid_forward():
    data = saved_data[5]
    t0 = data[0]
    gt = data[1]
    student = activation.Sigmoid()
    student(t0)
    closeness_test(student.state, gt, "sigmoid.state")
    print('sigmoid forward done!')


def test_sigmoid_derivative():
    data = saved_data[6]
    t0 = data[0]
    gt = data[1]
    student = activation.Sigmoid()
    student(t0)
    closeness_test(student.derivative(), gt, "sigmoid.derivative()")
    print('sigmoid derivative done!')


def test_tanh_forward():
    data = saved_data[9]
    t0 = data[0]
    gt = data[1]
    student = activation.Tanh()
    student(t0)
    closeness_test(student.state, gt, "tanh.state")
    print('tanh forward done!')


def test_tanh_derivative():
    data = saved_data[10]
    t0 = data[0]
    gt = data[1]
    student = activation.Tanh()
    student(t0)
    closeness_test(student.derivative(), gt, "tanh.derivative()")
    print('tanh derivative done!')


def reset_prng():
    np.random.seed(11785)


def weight_init(x, y):
    return np.random.randn(x, y)


def bias_init(x):
    return np.zeros((1, x))


def test_linear_layer_forward():
    data = saved_data[22]
    assert len(data) == 2
    x = data[0]
    gt = data[1]

    reset_prng()
    x = np.random.randn(20, 784)
    reset_prng()
    linear_layer = linear.Linear(784, 10, weight_init, bias_init)
    pred = linear_layer.forward(x)
    closeness_test(pred, gt, "linear_layer.forward(x)")
    print('linear layer forward done!')


def test_linear_layer_backward():
    data = saved_data[23]
    assert len(data) == 4
    x = data[0]
    y = data[1]
    soldW = data[2]
    soldb = data[3]

    reset_prng()
    linear_layer = linear.Linear(784, 10, weight_init, bias_init)
    linear_layer.forward(x)
    linear_layer.backward(y)

    closeness_test(linear_layer.dW, soldW, "linear_layer.dW")
    closeness_test(linear_layer.db, soldb, "linear_layer.db")
    print('linear layer backward done!')

# test_softmax_cross_entropy_forward()
# test_softmax_cross_entropy_derivative()
# test_relu_forward()
# test_relu_derivative()
# test_sigmoid_forward()
# test_sigmoid_derivative()
# test_tanh_forward()
# test_tanh_derivative()
# test_linear_layer_forward()
# test_linear_layer_backward()
