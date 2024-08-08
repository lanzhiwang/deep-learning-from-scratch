# coding: utf-8
import sys, os

sys.path.append(os.pardir)

import numpy as np
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

# 读入数据
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True,
                                                  one_hot_label=True)
# print("x_train:", x_train.shape)  # x_train: (60000, 784)
# print("t_train:", t_train.shape)  # t_train: (60000, 10)
# print("x_test:", x_test.shape)  # x_test: (10000, 784)
# print("t_test:", t_test.shape)  # t_test: (10000, 10)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

# 超参数
iters_num = 10000  # 适当设定循环的次数
train_size = x_train.shape[0]  # 60000
batch_size = 100
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)
# print("iter_per_epoch:", iter_per_epoch)  # iter_per_epoch: 600.0

for i in range(iters_num):
    # print("i:", i)  # 0 - 9999
    """
    >>> np.random.choice(60000, 10)
    array([ 8013, 14666, 58210, 23832, 52091, 10153, 8107, 19410, 27260, 21411])
    >>> np.random.choice(60000, 100)
    从 60000 条数据中每次取出 100 条数据
    """
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    # print("x_batch:", x_batch.shape)  # x_batch: (100, 784)
    # print("t_batch:", t_batch.shape)  # t_batch: (100, 10)

    # 梯度
    #grad = network.numerical_gradient(x_batch, t_batch)
    grad = network.gradient(x_batch, t_batch)

    # 更新
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(train_acc, test_acc)
