# coding: utf-8
import numpy as np


# 恒等函数
def identity_function(x):
    return x


# 阶跃函数
def step_function(x):
    return np.array(x > 0, dtype=np.int)


# sigmoid 函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)


# ReLU 函数
def relu(x):
    return np.maximum(0, x)


def relu_grad(x):
    grad = np.zeros(x)
    grad[x >= 0] = 1
    return grad


# softmax 函数
def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x)  # 溢出对策
    return np.exp(x) / np.sum(np.exp(x))


# 均方误差
def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t)**2)


# >>> import numpy as np
# >>> y = np.array([0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0])
# >>> t = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
# >>> y - t
# array([ 0.1 ,  0.05, -0.4 ,  0.  ,  0.05,  0.1 ,  0.  ,  0.1 ,  0.  ,
#         0.  ])
# >>> (y - t) ** 2
# array([0.01  , 0.0025, 0.16  , 0.    , 0.0025, 0.01  , 0.    , 0.01  ,
#        0.    , 0.    ])
# >>> np.sum((y - t)**2)
# 0.19500000000000006
# >>> 0.5 * np.sum((y - t)**2)
# 0.09750000000000003
# >>>

# 交叉熵误差
# >>> import numpy as np
# >>> y = np.array([0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0])
# >>> t = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
# >>> np.log(y)
# <stdin>:1: RuntimeWarning: divide by zero encountered in log
# array([-2.30258509, -2.99573227, -0.51082562,        -inf, -2.99573227,
#        -2.30258509,        -inf, -2.30258509,        -inf,        -inf])
# >>> t * np.log(y)
# <stdin>:1: RuntimeWarning: invalid value encountered in multiply
# array([-0.        , -0.        , -0.51082562,         nan, -0.        ,
#        -0.        ,         nan, -0.        ,         nan,         nan])
# >>> np.sum(t * np.log(y))
# nan
# >>> -np.sum(t * np.log(y))
# nan
# >>>

# def cross_entropy_error(y, t):
#     delta = 1e-7
#     return -np.sum(t * np.log(y + delta))


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 监督数据是one-hot-vector的情况下，转换为正确解标签的索引
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


# def cross_entropy_error(y, t):
#     if y.ndim == 1:
#         t = t.reshape(1, t.size)
#         y = y.reshape(1, y.size)
#     batch_size = y.shape[0]
#     return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


def softmax_loss(X, t):
    y = softmax(X)
    return cross_entropy_error(y, t)
