# coding: utf-8
import sys, os

sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import numpy as np
import pickle
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax


def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True,
                                                      flatten=True,
                                                      one_hot_label=False)
    return x_test, t_test


def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network


def predict(network, x):
    # print("predict x:", x.shape)
    # predict x: (784,)

    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    # print("predict a1:", a1.shape)
    # predict a1: (50,)

    z1 = sigmoid(a1)
    # print("predict z1:", z1.shape)
    # predict z1: (50,)

    a2 = np.dot(z1, W2) + b2
    # print("predict a2:", a2.shape)
    # predict a2: (100,)

    z2 = sigmoid(a2)
    # print("predict z2:", z2.shape)
    # predict z2: (100,)

    a3 = np.dot(z2, W3) + b3
    # print("predict a3:", a3.shape)
    # predict a3: (10,)

    y = softmax(a3)
    # print("predict y:", y.shape)
    # predict y: (10,)

    return y


x, t = get_data()
# print("x:", x.shape)  # x: (10000, 784)
# print("t:", t.shape)  # t: (10000,)

network = init_network()
# print("network['W1']:", network['W1'].shape)  # network['W1']: (784, 50)
# print("network['W2']:", network['W2'].shape)  # network['W2']: (50, 100)
# print("network['W3']:", network['W3'].shape)  # network['W3']: (100, 10)
# print("network['b1']:", network['b1'].shape)  # network['b1']: (50,)
# print("network['b2']:", network['b2'].shape)  # network['b2']: (100,)
# print("network['b3']:", network['b3'].shape)  # network['b3']: (10,)

accuracy_cnt = 0
for i in range(len(x)):
    # print("i:", i)  # i: 6716
    # print("x[i]:", x[i])
    # x[i]: [0.         0.         0.         0.         0.         0.
    #     0.         0.         0.         0.         0.         0.
    #     ...
    #     0.         0.         0.         0.         0.         0.
    #     0.         0.         0.         0.        ]
    # print("t[i]:", t[i])  # t[i]: 5
    y = predict(network, x[i])
    # print("y:", y)
    # y: [1.0158996e-02 1.5864036e-03 1.3458567e-02 7.0675765e-03 1.2783876e-02
    #  8.8168967e-01 1.0356861e-02 2.6394874e-03 5.9721325e-02 5.3730200e-04]
    p = np.argmax(y)  # 获取概率最高的元素的索引
    # print("p:", p)
    # p: 5
    if p == t[i]:
        accuracy_cnt += 1

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
