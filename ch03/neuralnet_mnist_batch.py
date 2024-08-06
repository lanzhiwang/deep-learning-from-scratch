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
    # predict x: (100, 784)

    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    # print("predict a1:", a1.shape)
    # predict a1: (100, 50)

    z1 = sigmoid(a1)
    # print("predict z1:", z1.shape)
    # predict z1: (100, 50)

    a2 = np.dot(z1, W2) + b2
    # print("predict a2:", a2.shape)
    # predict a2: (100, 100)

    z2 = sigmoid(a2)
    # print("predict z2:", z2.shape)
    # predict z2: (100, 100)

    a3 = np.dot(z2, W3) + b3
    # print("predict a3:", a3.shape)
    # predict a3: (100, 10)

    y = softmax(a3)
    # print("predict y:", y.shape)
    # predict y: (100, 10)

    return y


x, t = get_data()
network = init_network()

batch_size = 100  # 批数量
accuracy_cnt = 0

print("len(x):", len(x))
print("range(len(x)):", list(range(len(x))))
print("range(0, len(x), batch_size):", list(range(0, len(x), batch_size)))
# len(x): 10000
# range(len(x)): [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, ...
# range(0, len(x), batch_size): [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000, 3100, 3200, 3300, 3400, 3500, 3600, 3700, 3800, 3900, 4000, 4100, 4200, 4300, 4400, 4500, 4600, 4700, 4800, 4900, 5000, 5100, 5200, 5300, 5400, 5500, 5600, 5700, 5800, 5900, 6000, 6100, 6200, 6300, 6400, 6500, 6600, 6700, 6800, 6900, 7000, 7100, 7200, 7300, 7400, 7500, 7600, 7700, 7800, 7900, 8000, 8100, 8200, 8300, 8400, 8500, 8600, 8700, 8800, 8900, 9000, 9100, 9200, 9300, 9400, 9500, 9600, 9700, 9800, 9900]

for i in range(0, len(x), batch_size):
    # print("i:", i)
    # i: 2600

    # print("i + batch_size:", i + batch_size)
    # i + batch_size: 2700

    x_batch = x[i:i + batch_size]
    # print("x_batch:", x_batch.shape)
    # x_batch: (100, 784)

    y_batch = predict(network, x_batch)
    # print("y_batch:", y_batch.shape)
    # y_batch: (100, 10)

    p = np.argmax(y_batch, axis=1)
    # print("p:", p)
    # p: [8 6 2 9 5 7 5 8 8 6 8 5 1 4 8 4 5 8 3 0 6 2 7 3 3 2 1 0 7 3 4 0 3 9 3 2 8
    #     9 0 3 8 0 7 6 5 4 7 3 0 0 8 6 2 5 1 1 0 0 4 4 0 1 2 3 2 7 7 8 5 2 8 7 6 9
    #     1 4 1 6 4 2 4 3 5 4 3 9 5 0 1 5 3 8 9 1 9 4 9 5 5 2]

    accuracy_cnt += np.sum(p == t[i:i + batch_size])

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
