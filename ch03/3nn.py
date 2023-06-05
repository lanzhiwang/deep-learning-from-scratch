# coding: utf-8

import sys, os

# >>> import sys, os
# >>> os.pardir
# '..'
# >>>
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import numpy as np
from common.functions import sigmoid


def identity_function(x):
    return x


def init_network():
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])
    return network


def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = identity_function(a3)
    return y


network = init_network()
print("network:", network)
# network:
# {
#     'W1': array(
#         [
#             [0.1, 0.3, 0.5],
#             [0.2, 0.4, 0.6]
#         ]
#     ),
#     'b1': array(
#         [0.1, 0.2, 0.3]
#     ),
#     'W2': array(
#         [
#             [0.1, 0.4],
#             [0.2, 0.5],
#             [0.3, 0.6]
#         ]
#     ),
#     'b2': array(
#         [0.1, 0.2]
#     ),
#     'W3': array(
#         [
#             [0.1, 0.3],
#             [0.2, 0.4]
#         ]
#     ),
#     'b3': array(
#         [0.1, 0.2]
#     )
# }

x = np.array([1.0, 0.5])
y = forward(network, x)
print(y)
# [0.31682708 0.69627909]

# >>> import numpy as np
# >>> A = np.array([[1,2], [3,4]])
# >>> print(A)
# [[1 2]
#  [3 4]]
# >>> B = np.array([[5,6], [7,8]])
# >>> print(B)
# [[5 6]
#  [7 8]]
# >>> np.dot(A, B)
# array([[19, 22],
#        [43, 50]])
# >>> A * B
# array([[ 5, 12],
#        [21, 32]])
# >>>

# >>> import numpy as np
# >>> X = np.array([1.0, 0.5])
# >>> X
# array([1. , 0.5])
# >>>
# >>> W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
# >>> W1
# array([[0.1, 0.3, 0.5],
#        [0.2, 0.4, 0.6]])
# >>> B1 = np.array([0.1, 0.2, 0.3])
# >>> np.dot(X, W1)
# array([0.2, 0.5, 0.8])
# >>> np.dot(X, W1) + B1
# array([0.3, 0.7, 1.1])
# >>>
