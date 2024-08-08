# coding: utf-8
import sys, os

sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
from common.functions import *
from common.gradient import numerical_gradient


class TwoLayerNet:

    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size,
                 weight_init_std=0.01):
        # network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
        # 初始化权重
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(
            input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(
            hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
        # print("self.params['W1']:", self.params['W1'].shape)  # self.params['W1']: (784, 50)
        # print("self.params['b1']:", self.params['b1'].shape)  # self.params['b1']: (50,)
        # print("self.params['W2']:", self.params['W2'].shape)  # self.params['W2']: (50, 10)
        # print("self.params['b2']:", self.params['b2'].shape)  # self.params['b2']: (10,)

    def predict(self, x):
        # print("x:", x.shape)  # x: (100, 784)

        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1) + b1  # (100, 784) * (784, 50) + (50,) = (100, 50)
        z1 = sigmoid(a1)  # (100, 50)
        a2 = np.dot(z1, W2) + b2  # (100, 50) * (50, 10) + (10,) = (100, 10)
        y = softmax(a2)  # (100, 10)

        return y

    # x:输入数据, t:监督数据
    def loss(self, x, t):
        # print("x:", x.shape)  # x: (100, 784)
        # print("t:", t.shape)  # t: (100, 10)

        y = self.predict(x)  # (100, 10)

        return cross_entropy_error(y, t)

    def accuracy(self, x, t):
        # print("x:", x.shape)  # x: (100, 784)
        # print("t:", t.shape)  # t: (100, 10)

        y = self.predict(x)  # (100, 10)
        """
        >>> import numpy as np
        >>> a = np.random.randint(0, 15, 15)
        >>> a
        array([ 3,  0, 11,  4,  1,  8, 10, 11, 13,  3,  8,  6,  8,  4,  0])
        >>> a = a.reshape(3, 5)
        >>> a
        array([[ 3,  0, 11,  4,  1],
               [ 8, 10, 11, 13,  3],
               [ 8,  6,  8,  4,  0]])
        >>> a = np.argmax(a, axis=1)
        >>> a
        array([2, 3, 0])
        >>>
        """
        y = np.argmax(y, axis=1)
        # print("y:", y.shape)  # y: (100,)
        t = np.argmax(t, axis=1)
        # print("t:", t.shape)  # t: (100,)
        """
        >>> a = np.random.randint(0, 15, 15)
        >>> a
        array([10,  6,  5,  4,  6, 12, 11,  0, 14, 12,  3,  1,  6,  2, 11])
        >>> b = np.random.randint(0, 15, 15)
        >>> b
        array([ 9, 11,  6,  8, 10,  8,  6,  3,  6,  3, 14,  4, 12, 12, 11])
        >>> a == b
        array([False, False, False, False, False, False, False, False, False,
               False, False, False, False, False,  True])
        >>> np.sum(a ==b)
        np.int64(1)
        >>>
        """
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    # x:输入数据, t:监督数据
    def numerical_gradient(self, x, t):
        # print("x:", x.shape)  # x: (100, 784)
        # print("t:", t.shape)  # t: (100, 10)

        loss_W = lambda W: self.loss(x, t)

        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads

    def gradient(self, x, t):
        # print("x:", x.shape)  # x: (100, 784)
        # print("t:", t.shape)  # t: (100, 10)

        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        # print("W1:", W1.shape)  # W1: (784, 50)
        # print("b1:", b1.shape)  # b1: (50,)
        # print("W2:", W2.shape)  # W2: (50, 10)
        # print("b2:", b2.shape)  # b2: (10,)

        grads = {}

        batch_num = x.shape[0]
        # print("batch_num:", batch_num)  # batch_num: 100

        # forward
        a1 = np.dot(x, W1) + b1
        # print("a1:", a1.shape) # a1: (100, 50)
        z1 = sigmoid(a1)
        # print("z1:", z1.shape)  # z1: (100, 50)
        a2 = np.dot(z1, W2) + b2
        # print("a2:", a2.shape)  # a2: (100, 10)
        y = softmax(a2)
        # print("y:", y.shape)  # y: (100, 10)

        # backward
        dy = (y - t) / batch_num
        # print("dy:", dy.shape)  # dy: (100, 10)
        grads['W2'] = np.dot(z1.T, dy)
        grads['b2'] = np.sum(dy, axis=0)
        # print("grads['W2']:", grads['W2'].shape)  # grads['W2']: (50, 10)
        # print("grads['b2']:", grads['b2'].shape)  # grads['b2']: (10,)

        da1 = np.dot(dy, W2.T)  # (100, 10) * (10, 50) = (100, 50)
        dz1 = sigmoid_grad(a1) * da1
        grads['W1'] = np.dot(x.T, dz1)  # (784, 100) * (100, 50) = (784, 50)
        grads['b1'] = np.sum(dz1, axis=0)
        # print("grads['W1']:", grads['W1'].shape)  # grads['W1']: (784, 50)
        # print("grads['b1']:", grads['b1'].shape)  # grads['b1']: (50,)

        return grads


if __name__ == '__main__':
    net = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
    print("net.params['W1']:",
          net.params['W1'].shape)  # net.params['W1']: (784, 50)
    print("net.params['b1']:",
          net.params['b1'].shape)  # net.params['b1']: (50,)
    print("net.params['W2']:",
          net.params['W2'].shape)  # net.params['W2']: (50, 10)
    print("net.params['b2']:",
          net.params['b2'].shape)  # net.params['b2']: (10,)

    x = np.random.rand(100, 784)  # 伪输入数据(100笔)
    y = net.predict(x)
    print("y:", y.shape)  # y: (100, 10)

    x = np.random.rand(100, 784)  # 伪输入数据(100笔)
    t = np.random.rand(100, 10)  # 伪正确解标签(100笔)
    loss = net.loss(x, t)
    print("loss:", loss)  # loss: 2.3080205824403346

    accuracy = net.accuracy(x, t)
    print("accuracy:", accuracy)  # accuracy: 0.09

    grads = net.numerical_gradient(x, t)
    print("grads['W1']:", grads['W1'].shape)  # grads['W1']: (784, 50)
    print("grads['b1']:", grads['b1'].shape)  # grads['b1']: (50,)
    print("grads['W2']:", grads['W2'].shape)  # grads['W2']: (50, 10)
    print("grads['b2']:", grads['b2'].shape)  # grads['b2']: (10,)
