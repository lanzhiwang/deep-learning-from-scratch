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
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)
    if i == 0:
        print("x:", x)
        # x: [0.         0.         0.         0.         0.         0.
        #     0.         0.         0.         0.         0.         0.
        #     ...
        #     0.4745098  0.99607843 0.8117647  0.07058824 0.         0.
        #     0.         0.         0.         0.         0.         0.
        #     0.         0.         0.         0.         0.         0.
        #     0.         0.         0.         0.         0.         0.
        #     0.         0.         0.         0.         0.         0.
        #     0.         0.         0.         0.         0.         0.
        #     0.         0.         0.         0.         0.         0.
        #     0.         0.         0.         0.        ]
        print("x:", x.shape)  # (784,)

        print("a1:", a1.shape)  # (50,)
        print("z1:", z1.shape)  # (50,)
        print("a2:", a2.shape)  # (100,)
        print("z2:", z2.shape)  # (100,)
        print("a3:", a3.shape)  # (10,)
        print("y:", y.shape)  # (10,)

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

# 识别精度(accuracy)
accuracy_cnt = 0
for i in range(len(x)):
    y = predict(network, x[i], i)
    # np.argmax(y) 将获取被赋给参数 y 的数组中的最大值元素的索引
    p = np.argmax(y)  # 获取概率最高的元素的索引
    if p == t[i]:
        accuracy_cnt += 1

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
