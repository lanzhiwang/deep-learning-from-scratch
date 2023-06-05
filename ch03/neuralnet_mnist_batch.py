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


def predict(network, x, i):
    w1, w2, w3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, w1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, w2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, w3) + b3
    y = softmax(a3)
    if i == 0:
        print("x:", x)
        # x: [[0. 0. 0. ... 0. 0. 0.]
        #     [0. 0. 0. ... 0. 0. 0.]
        #     [0. 0. 0. ... 0. 0. 0.]
        #     ...
        #     [0. 0. 0. ... 0. 0. 0.]
        #     [0. 0. 0. ... 0. 0. 0.]
        #     [0. 0. 0. ... 0. 0. 0.]]
        print("x:", x.shape)  # (100, 784)

        print("a1:", a1.shape)  # (100, 50)
        print("z1:", z1.shape)  # (100, 50)
        print("a2:", a2.shape)  # (100, 100)
        print("z2:", z2.shape)  # (100, 100)
        print("a3:", a3.shape)  # (100, 10)
        print("y:", y.shape)  # (100, 10)

    return y


x, t = get_data()
print(x.shape)  # (10000, 784)
print(t.shape)  # (10000,)
print(len(x))  # 10000

network = init_network()
print(network['W1'].shape, network['W2'].shape, network['W3'].shape)
# (784, 50) (50, 100) (100, 10)
print(network['b1'].shape, network['b2'].shape, network['b3'].shape)
# (50,) (100,) (10,)

batch_size = 100  # 批数量
accuracy_cnt = 0

# >>> for i in range(0, 10000, 100):
# ...     print(i, i + 100)
# ...
# 0 100
# 100 200
# 200 300
# 300 400
# 400 500
# 500 600
# ...
# 9700 9800
# 9800 9900
# 9900 10000
# >>>

for i in range(0, len(x), batch_size):
    x_batch = x[i:i + batch_size]
    y_batch = predict(network, x_batch, i)
    p = np.argmax(y_batch, axis=1)
    # print(np.argmax(y, axis=1))
    # [7 2 1 0 4 1 4 9 6 9 0 6 9 0 1 5 9 7 3 4 9 6 6 5 4 0 7 4 0 1 3 1 3 6 7 2 7
    #     1 2 1 1 7 4 2 3 5 1 2 4 4 6 3 5 5 6 0 4 1 9 5 7 8 9 3 7 4 2 4 3 0 7 0 2 9
    #     1 7 3 2 9 7 7 6 2 7 8 4 7 3 6 1 3 6 4 3 1 4 1 7 6 9]
    accuracy_cnt += np.sum(p == t[i:i + batch_size])

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
