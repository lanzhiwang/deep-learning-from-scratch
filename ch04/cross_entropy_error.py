# coding: utf-8
import numpy as np


def cross_entropy_error(y, t):
    print("y:", y)
    print("y.ndim:", y.ndim)
    print("y.size:", y.size)
    print("y.shape:", y.shape)
    print("t:", t)
    print("t.ndim:", t.ndim)
    print("t.size:", t.size)
    print("t.shape:", t.shape)
    print("*********************************************")

    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        print("y:", y)
        print("y.ndim:", y.ndim)
        print("y.size:", y.size)
        print("y.shape:", y.shape)
        print("t:", t)
        print("t.ndim:", t.ndim)
        print("t.size:", t.size)
        print("t.shape:", t.shape)
        print("*********************************************")

    # 监督数据是one-hot-vector的情况下，转换为正确解标签的索引
    if t.size == y.size:
        t = t.argmax(axis=1)
        print("y:", y)
        print("y.ndim:", y.ndim)
        print("y.size:", y.size)
        print("y.shape:", y.shape)
        print("t:", t)
        print("t.ndim:", t.ndim)
        print("t.size:", t.size)
        print("t.shape:", t.shape)
        print("*********************************************")

    batch_size = y.shape[0]
    print("batch_size:", batch_size)
    a = np.arange(batch_size)
    print("a:", a)
    l = [np.arange(batch_size), t]
    print("l:", l)
    y_new = y[np.arange(batch_size), t]
    print("y_new:", y_new)
    print("y_new.ndim:", y_new.ndim)
    print("y_new.size:", y_new.size)
    print("y_new.shape:", y_new.shape)

    result = -np.sum(np.log(y_new + 1e-7)) / batch_size
    print("result:", result)

    return result


# y = np.array([
#     [1, 2, 3],
#     [4, 5, 6],
#     [7, 8, 9],
#     [10, 11, 12],
#     [13, 14, 15]
# ])
# t = np.array([1, 1, 0, 2, 1])
# cross_entropy_error(y, t)
# y: [[ 1  2  3]
#  [ 4  5  6]
#  [ 7  8  9]
#  [10 11 12]
#  [13 14 15]]
# y.ndim: 2
# y.size: 15
# y.shape: (5, 3)
# t: [1 1 0 2 1]
# t.ndim: 1
# t.size: 5
# t.shape: (5,)
# *********************************************
# batch_size: 5
# a: [0 1 2 3 4]
# l: [array([0, 1, 2, 3, 4]), array([1, 1, 0, 2, 1])]
# y_new: [ 2  5  7 12 14]
# y_new.ndim: 1
# y_new.size: 5
# y_new.shape: (5,)
# result: -1.8744918642429043

# y = np.array([0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0])
# t = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
# cross_entropy_error(y, t)
# y: [0.1  0.05 0.6  0.   0.05 0.1  0.   0.1  0.   0.  ]
# y.ndim: 1
# y.size: 10
# y.shape: (10,)
# t: [0 0 1 0 0 0 0 0 0 0]
# t.ndim: 1
# t.size: 10
# t.shape: (10,)
# *********************************************
# y: [[0.1  0.05 0.6  0.   0.05 0.1  0.   0.1  0.   0.  ]]
# y.ndim: 2
# y.size: 10
# y.shape: (1, 10)
# t: [[0 0 1 0 0 0 0 0 0 0]]
# t.ndim: 2
# t.size: 10
# t.shape: (1, 10)
# *********************************************
# y: [[0.1  0.05 0.6  0.   0.05 0.1  0.   0.1  0.   0.  ]]
# y.ndim: 2
# y.size: 10
# y.shape: (1, 10)
# t: [2]
# t.ndim: 1
# t.size: 1
# t.shape: (1,)
# *********************************************
# batch_size: 1
# a: [0]
# l: [array([0]), array([2])]
# y_new: [0.6]
# y_new.ndim: 1
# y_new.size: 1
# y_new.shape: (1,)
# result: 0.510825457099338

# y = np.array([
#     [1, 2, 3],
#     [4, 5, 6],
#     [7, 8, 9],
#     [10, 11, 12],
#     [13, 14, 15]
# ])
# t = np.array([
#     [0, 1, 0],
#     [1, 0, 0],
#     [0, 1, 0],
#     [0, 0, 1],
#     [0, 0, 1]
# ])
# cross_entropy_error(y, t)
# y: [[ 1  2  3]
#  [ 4  5  6]
#  [ 7  8  9]
#  [10 11 12]
#  [13 14 15]]
# y.ndim: 2
# y.size: 15
# y.shape: (5, 3)
# t: [[0 1 0]
#  [1 0 0]
#  [0 1 0]
#  [0 0 1]
#  [0 0 1]]
# t.ndim: 2
# t.size: 15
# t.shape: (5, 3)
# *********************************************
# y: [[ 1  2  3]
#  [ 4  5  6]
#  [ 7  8  9]
#  [10 11 12]
#  [13 14 15]]
# y.ndim: 2
# y.size: 15
# y.shape: (5, 3)
# t: [1 0 1 2 2]
# t.ndim: 1
# t.size: 5
# t.shape: (5,)
# *********************************************
# batch_size: 5
# a: [0 1 2 3 4]
# l: [array([0, 1, 2, 3, 4]), array([1, 0, 1, 2, 2])]
# y_new: [ 2  4  8 12 15]
# y_new.ndim: 1
# y_new.size: 5
# y_new.shape: (5,)
# result: -1.8703680073499762
