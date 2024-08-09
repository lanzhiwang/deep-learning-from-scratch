# coding: utf-8
import sys, os

sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import numpy as np
from collections import OrderedDict
from common.layers import *
from common.gradient import numerical_gradient


class MultiLayerNet:
    """全连接的多层神经网络

    Parameters
    ----------
    input_size: 输入大小(MNIST 的情况下为 784)
    hidden_size_list: 隐藏层的神经元数量的列表(e.g. [100, 100, 100])
    output_size: 输出大小(MNIST 的情况下为 10)
    activation: 'relu' or 'sigmoid'
    weight_init_std: 指定权重的标准差(e.g. 0.01)
        指定 'relu' 或 'he' 的情况下设定 "He 的初始值"
        指定 'sigmoid' 或 'xavier' 的情况下设定 "Xavier 的初始值"
    weight_decay_lambda: Weight Decay(L2 范数)的强度
    """

    # MultiLayerNet(input_size=784, hidden_size_list=[100, 100, 100, 100], output_size=10)
    def __init__(self,
                 input_size,
                 hidden_size_list,
                 output_size,
                 activation='relu',
                 weight_init_std='relu',
                 weight_decay_lambda=0):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size_list = hidden_size_list
        self.hidden_layer_num = len(hidden_size_list)
        self.weight_decay_lambda = weight_decay_lambda
        self.params = {}
        # print("self.input_size:", self.input_size)  # self.input_size: 784
        # print("self.output_size:", self.output_size)  # self.output_size: 10
        # print("self.hidden_size_list:", self.hidden_size_list)  # self.hidden_size_list: [100, 100, 100, 100]
        # print("self.hidden_layer_num:", self.hidden_layer_num)  # self.hidden_layer_num: 4
        # print("self.weight_decay_lambda:", self.weight_decay_lambda)  # self.weight_decay_lambda: 0
        # print("self.params:", self.params)  # self.params: {}

        # 初始化权重
        self.__init_weight(weight_init_std)
        # for key in self.params.keys():
        #     print("self.params", key, ":", self.params[key].shape)
        #     self.params W1 : (784, 100)
        #     self.params b1 : (100,)
        #     self.params W2 : (100, 100)
        #     self.params b2 : (100,)
        #     self.params W3 : (100, 100)
        #     self.params b3 : (100,)
        #     self.params W4 : (100, 100)
        #     self.params b4 : (100,)
        #     self.params W5 : (100, 10)
        #     self.params b5 : (10,)

        # 生成层
        activation_layer = {'sigmoid': Sigmoid, 'relu': Relu}
        self.layers = OrderedDict()
        for idx in range(1, self.hidden_layer_num + 1):
            # print("idx:", idx)  # idx: 1 - 4
            self.layers['Affine' + str(idx)] = Affine(
                self.params['W' + str(idx)], self.params['b' + str(idx)])
            self.layers['Activation_function' +
                        str(idx)] = activation_layer[activation]()
        # print("self.layers:", self.layers)
        # self.layers: OrderedDict(
        #     [
        #         ('Affine1', <common.layers.Affine object at 0x73cdf090ad10>),
        #         ('Activation_function1', <common.layers.Relu object at 0x73cdf090ace0>),
        #         ('Affine2', <common.layers.Affine object at 0x73cdf090ada0>),
        #         ('Activation_function2', <common.layers.Relu object at 0x73cdf090bbb0>),
        #         ('Affine3', <common.layers.Affine object at 0x73cdf090b820>),
        #         ('Activation_function3', <common.layers.Relu object at 0x73cdf090b7f0>),
        #         ('Affine4', <common.layers.Affine object at 0x73cdf0911600>),
        #         ('Activation_function4', <common.layers.Relu object at 0x73cdf0911630>)
        #     ]
        # )

        idx = self.hidden_layer_num + 1
        self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(idx)],
                                                  self.params['b' + str(idx)])
        # print("self.layers:", self.layers)
        # self.layers: OrderedDict(
        #     [
        #         ('Affine1', <common.layers.Affine object at 0x73cdf090ad10>),
        #         ('Activation_function1', <common.layers.Relu object at 0x73cdf090ace0>),
        #         ('Affine2', <common.layers.Affine object at 0x73cdf090ada0>),
        #         ('Activation_function2', <common.layers.Relu object at 0x73cdf090bbb0>),
        #         ('Affine3', <common.layers.Affine object at 0x73cdf090b820>),
        #         ('Activation_function3', <common.layers.Relu object at 0x73cdf090b7f0>),
        #         ('Affine4', <common.layers.Affine object at 0x73cdf0911600>),
        #         ('Activation_function4', <common.layers.Relu object at 0x73cdf0911630>),
        #         ('Affine5', <common.layers.Affine object at 0x73cdf090ad40>)
        #     ]
        # )

        self.last_layer = SoftmaxWithLoss()

    def __init_weight(self, weight_init_std):
        """设定权重的初始值

        Parameters
        ----------
        weight_init_std: 指定权重的标准差(e.g. 0.01)
            指定 'relu' 或 'he' 的情况下设定 "He 的初始值"
            指定 'sigmoid' 或 'xavier' 的情况下设定 "Xavier 的初始值"
        """
        all_size_list = [self.input_size
                         ] + self.hidden_size_list + [self.output_size]
        # print("all_size_list:", all_size_list)  # all_size_list: [784, 100, 100, 100, 100, 10]

        for idx in range(1, len(all_size_list)):
            # print("idx:", idx)
            scale = weight_init_std
            if str(weight_init_std).lower() in ('relu', 'he'):
                scale = np.sqrt(2.0 /
                                all_size_list[idx - 1])  # 使用 ReLU 的情况下推荐的初始值

            elif str(weight_init_std).lower() in ('sigmoid', 'xavier'):
                scale = np.sqrt(
                    1.0 / all_size_list[idx - 1])  # 使用 sigmoid 的情况下推荐的初始值
            # print("scale:", scale)
            # idx: 1
            # scale: 0.050507627227610534
            # idx: 2
            # scale: 0.1414213562373095
            # idx: 3
            # scale: 0.1414213562373095
            # idx: 4
            # scale: 0.1414213562373095
            # idx: 5
            # scale: 0.1414213562373095

            self.params['W' + str(idx)] = scale * np.random.randn(
                all_size_list[idx - 1], all_size_list[idx])
            self.params['b' + str(idx)] = np.zeros(all_size_list[idx])
            # print("self.params keys:", self.params.keys())
            # self.params keys: dict_keys(['W1', 'b1'])
            # self.params keys: dict_keys(['W1', 'b1', 'W2', 'b2'])
            # self.params keys: dict_keys(['W1', 'b1', 'W2', 'b2', 'W3', 'b3'])
            # self.params keys: dict_keys(['W1', 'b1', 'W2', 'b2', 'W3', 'b3', 'W4', 'b4'])
            # self.params keys: dict_keys(['W1', 'b1', 'W2', 'b2', 'W3', 'b3', 'W4', 'b4', 'W5', 'b5'])

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self, x, t):
        """
        求损失函数

        损失函数的值
        """
        y = self.predict(x)

        weight_decay = 0
        for idx in range(1, self.hidden_layer_num + 2):
            W = self.params['W' + str(idx)]
            weight_decay += 0.5 * self.weight_decay_lambda * np.sum(W**2)

        return self.last_layer.forward(y, t) + weight_decay

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        """
        求梯度(数值微分)

        具有各层的梯度的字典变量
            grads['W1']、grads['W2']、...是各层的权重
            grads['b1']、grads['b2']、...是各层的偏置
        """
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        for idx in range(1, self.hidden_layer_num + 2):
            grads['W' + str(idx)] = numerical_gradient(
                loss_W, self.params['W' + str(idx)])
            grads['b' + str(idx)] = numerical_gradient(
                loss_W, self.params['b' + str(idx)])

        return grads

    def gradient(self, x, t):
        """
        求梯度(误差反向传播法)

        具有各层的梯度的字典变量
            grads['W1']、grads['W2']、...是各层的权重
            grads['b1']、grads['b2']、...是各层的偏置
        """
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 设定
        grads = {}
        for idx in range(1, self.hidden_layer_num + 2):
            grads['W' + str(idx)] = self.layers['Affine' + str(
                idx)].dW + self.weight_decay_lambda * self.layers['Affine' +
                                                                  str(idx)].W
            grads['b' + str(idx)] = self.layers['Affine' + str(idx)].db

        return grads
