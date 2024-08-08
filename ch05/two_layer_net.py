# coding: utf-8
import sys, os

sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import numpy as np
from common.layers import *
from common.gradient import numerical_gradient
from collections import OrderedDict


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

        # 生成层
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])

        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x):
        # print("x:", x.shape)  # x: (100, 784)
        for layer in self.layers.values():
            # print("predict layer:", layer)
            x = layer.forward(x)
            # print("x:", x.shape)
            # predict layer: <common.layers.Affine object at 0x73672e0ae500>
            # x: (100, 50)
            # predict layer: <common.layers.Relu object at 0x73672e0ae440>
            # x: (100, 50)
            # predict layer: <common.layers.Affine object at 0x73672e0aecb0>
            # x: (100, 10)

        return x

    # x:输入数据, t:监督数据
    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
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
        if t.ndim != 1:
            t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    # x:输入数据, t:监督数据
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads

    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 设定
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers[
            'Affine1'].db
        grads['W2'], grads['b2'] = self.layers['Affine2'].dW, self.layers[
            'Affine2'].db

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

    grads = net.gradient(x, t)
    print("grads['W1']:", grads['W1'].shape)  # grads['W1']: (784, 50)
    print("grads['b1']:", grads['b1'].shape)  # grads['b1']: (50,)
    print("grads['W2']:", grads['W2'].shape)  # grads['W2']: (50, 10)
    print("grads['b2']:", grads['b2'].shape)  # grads['b2']: (10,)
