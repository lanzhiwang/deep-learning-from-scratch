# coding: utf-8
import numpy as np


def identity_function(x):
    """
    恒等函数
    >>> import numpy as np
    >>> from common.functions import identity_function
    >>> x = np.array([-1.0, 1.0, 2.0])
    >>> y = identity_function(x)
    >>> y
    array([-1.,  1.,  2.])
    >>>
    """
    return x


def step_function(x):
    """
    阶跃函数
    >>> import numpy as np
    >>> x = np.array([-1.0, 1.0, 2.0])
    >>> x
    array([-1.,  1.,  2.])
    >>> y = x > 0
    >>> y
    array([False,  True,  True])
    >>> y = y.astype(np.int32)
    >>> y
    array([0, 1, 1], dtype=int32)
    >>>
    """
    return np.array(x > 0, dtype=np.int32)


def sigmoid(x):
    """
    sigmoid 函数
    >>> import numpy as np
    >>> x = np.arange(-5.0, 5.0, 1.0)
    >>> x
    array([-5., -4., -3., -2., -1.,  0.,  1.,  2.,  3.,  4.])
    >>> np.exp(-x)
    array([1.48413159e+02, 5.45981500e+01, 2.00855369e+01, 7.38905610e+00,
           2.71828183e+00, 1.00000000e+00, 3.67879441e-01, 1.35335283e-01,
           4.97870684e-02, 1.83156389e-02])
    >>> 1 + np.exp(-x)
    array([149.4131591 ,  55.59815003,  21.08553692,   8.3890561 ,
             3.71828183,   2.        ,   1.36787944,   1.13533528,
             1.04978707,   1.01831564])
    >>> 1 / (1 + np.exp(-x))
    array([0.00669285, 0.01798621, 0.04742587, 0.11920292, 0.26894142,
           0.5       , 0.73105858, 0.88079708, 0.95257413, 0.98201379])
    >>>
    """
    return 1 / (1 + np.exp(-x))


def relu(x):
    """
    ReLU 函数
    >>> import numpy as np
    >>> x = np.arange(-5.0, 5.0, 1.0)
    >>> x
    array([-5., -4., -3., -2., -1.,  0.,  1.,  2.,  3.,  4.])
    >>> np.maximum(0, x)
    array([0., 0., 0., 0., 0., 0., 1., 2., 3., 4.])
    >>>
    """
    return np.maximum(0, x)


def softmax(x):
    """
    softmax 函数
    >>> import numpy as np
    >>> a = np.array([0.3, 2.9, 4.0])
    >>> exp_a = np.exp(a)  # 指数函数
    >>> exp_a
    array([ 1.34985881, 18.17414537, 54.59815003])
    >>> sum_exp_a = np.sum(exp_a)  # 指数函数的和
    >>> sum_exp_a
    np.float64(74.1221542101633)
    >>> y = exp_a / sum_exp_a
    >>> y
    array([0.01821127, 0.24519181, 0.73659691])
    >>>
    """
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x)  # 溢出对策
    return np.exp(x) / np.sum(np.exp(x))


def mean_squared_error(y, t):
    """
    均方误差
    >>> # 设"2"为正确解
    >>> t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0] >>>
    >>> # 例1: "2" 的概率最高的情况(0.6)
    >>> y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
    >>> mean_squared_error(np.array(y), np.array(t))
    0.097500000000000031
    >>>
    >>> # 例2: "7" 的概率最高的情况(0.6)
    >>> y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
    >>> mean_squared_error(np.array(y), np.array(t))
    0.59750000000000003
    """
    return 0.5 * np.sum((y - t)**2)


"""
交叉熵误差
>>> t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
>>> y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
>>> cross_entropy_error(np.array(y), np.array(t))
0.51082545709933802
>>>
>>> y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
>>> cross_entropy_error(np.array(y), np.array(t))
2.3025840929945458

def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))
"""

##############################################################
"""
交叉熵误差
>>> import numpy as np
>>> t1 = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
>>> t2 = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
>>> y1 = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
>>> y2 = [0.1, 0.05, 0.0, 0.0, 0.05, 0.1, 0.0, 0.1, 0.6, 0.0]
>>> y = np.array([y1, y2])
>>> t = np.array([t1, t2])
>>> y
array([[0.1 , 0.05, 0.6 , 0.  , 0.05, 0.1 , 0.  , 0.1 , 0.  , 0.  ],
       [0.1 , 0.05, 0.  , 0.  , 0.05, 0.1 , 0.  , 0.1 , 0.6 , 0.  ]])
>>> y.shape
(2, 10)
>>> y.ndim
2
>>> t
array([[0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]])
>>> t.shape
(2, 10)
>>> t.ndim
2
>>> batch_size = y.shape[0]
>>> batch_size
2
>>> np.log(y + 1e-7)
array([[ -2.30258409,  -2.99573027,  -0.51082546, -16.11809565,
         -2.99573027,  -2.30258409, -16.11809565,  -2.30258409,
        -16.11809565, -16.11809565],
       [ -2.30258409,  -2.99573027, -16.11809565, -16.11809565,
         -2.99573027,  -2.30258409, -16.11809565,  -2.30258409,
         -0.51082546, -16.11809565]])
>>> t * np.log(y + 1e-7)
array([[-0.        , -0.        , -0.51082546, -0.        , -0.        ,
        -0.        , -0.        , -0.        , -0.        , -0.        ],
       [-0.        , -0.        , -0.        , -0.        , -0.        ,
        -0.        , -0.        , -0.        , -0.51082546, -0.        ]])
>>> np.sum(t * np.log(y + 1e-7))
np.float64(-1.021650914198676)
>>> -np.sum(t * np.log(y + 1e-7)) / batch_size
np.float64(0.510825457099338)
>>>
>>> y = np.array(y1)
>>> t = np.array(t1)
>>> y
array([0.1 , 0.05, 0.6 , 0.  , 0.05, 0.1 , 0.  , 0.1 , 0.  , 0.  ])
>>> y.shape
(10,)
>>> y.ndim
1
>>> y.size
10
>>> t
array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
>>> t.shape
(10,)
>>> t.ndim
1
>>> t.size
10
>>> t = t.reshape(1, t.size)
>>> t
array([[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]])
>>> t.shape
(1, 10)
>>> y = y.reshape(1, y.size)
>>> y.shape
(1, 10)
>>>

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size
"""

##############################################################


def cross_entropy_error(y, t):
    """
    交叉熵误差
    >>> import numpy as np
    >>> t1 = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    >>> t2 = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
    >>> y1 = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
    >>> y2 = [0.1, 0.05, 0.0, 0.0, 0.05, 0.1, 0.0, 0.1, 0.6, 0.0]
    >>> y = np.array([y1, y2])
    >>> t = np.array([t1, t2])
    >>> y
    array([[0.1 , 0.05, 0.6 , 0.  , 0.05, 0.1 , 0.  , 0.1 , 0.  , 0.  ],
           [0.1 , 0.05, 0.  , 0.  , 0.05, 0.1 , 0.  , 0.1 , 0.6 , 0.  ]])
    >>> t
    array([[0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]])
    >>> y.size
    20
    >>> t.size
    20
    >>> t = t.argmax(axis=1)
    >>> t
    array([2, 8])
    >>> t.shape
    (2,)
    >>> batch_size = y.shape[0]
    >>> batch_size
    2
    >>> np.arange(batch_size)
    array([0, 1])
    >>> y[np.arange(batch_size), t]
    array([0.6, 0.6])
    >>> np.log(y[np.arange(batch_size), t] + 1e-7)
    array([-0.51082546, -0.51082546])
    >>> -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
    np.float64(0.510825457099338)
    >>>
    作为参考, 简单介绍一下 np.log(y[np.arange(batch_size), t]).
    np.arange(batch_size) 会生成一个从 0 到 batch_size-1 的数组.
    比如当 batch_size 为 5 时, np.arange(batch_size) 会生成一个 NumPy 数组 [0, 1, 2, 3, 4].
    因为 t 中标签是以 [2, 7, 0, 9, 4] 的形式存储的, 所以 y[np.arange(batch_size), t] 能抽出各个数据的正确解标签对应的神经网络的输出
    (在这个例子中, y[np.arange(batch_size), t] 会生成 NumPy 数组 [y[0,2], y[1,7], y[2,0], y[3,9], y[4,4]]).
    ##############################################################
    >>> import numpy as np
    >>> y1 = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
    >>> y2 = [0.1, 0.05, 0.0, 0.0, 0.05, 0.1, 0.0, 0.1, 0.6, 0.0]
    >>> y = np.array([y1, y2])
    >>> t = np.array([2, 8])
    >>> y
    array([[0.1 , 0.05, 0.6 , 0.  , 0.05, 0.1 , 0.  , 0.1 , 0.  , 0.  ],
           [0.1 , 0.05, 0.  , 0.  , 0.05, 0.1 , 0.  , 0.1 , 0.6 , 0.  ]])
    >>> t
    array([2, 8])
    >>> y.ndim
    2
    >>> y.size
    20
    >>> t.size
    2
    >>> y.shape
    (2, 10)
    >>> batch_size = y.shape[0]
    >>> batch_size
    2
    >>> np.arange(batch_size)
    array([0, 1])
    >>> y[np.arange(batch_size), t]
    array([0.6, 0.6])
    >>> -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
    np.float64(0.510825457099338)
    >>>
    """
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 监督数据是one-hot-vector的情况下，转换为正确解标签的索引
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)


def relu_grad(x):
    grad = np.zeros(x)
    grad[x >= 0] = 1
    return grad


def softmax_loss(X, t):
    y = softmax(X)
    return cross_entropy_error(y, t)
