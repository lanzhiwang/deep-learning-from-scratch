# coding: utf-8
import sys, os

sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import numpy as np
from collections import OrderedDict
from common.layers import *
from common.gradient import numerical_gradient


class MultiLayerNetExtend:
    """扩展版的全连接的多层神经网络

    具有Weiht Decay、Dropout、Batch Normalization的功能

    Parameters
    ----------
    input_size : 输入大小(MNIST的情况下为784)
    hidden_size_list : 隐藏层的神经元数量的列表(e.g. [100, 100, 100])
    output_size : 输出大小(MNIST的情况下为10)
    activation : 'relu' or 'sigmoid'
    weight_init_std : 指定权重的标准差(e.g. 0.01)
        指定'relu'或'he'的情况下设定“He的初始值”
        指定'sigmoid'或'xavier'的情况下设定“Xavier的初始值”
    weight_decay_lambda : Weight Decay(L2范数)的强度
    use_dropout: 是否使用Dropout
    dropout_ration : Dropout的比例
    use_batchNorm: 是否使用Batch Normalization
    """

    def __init__(self,
                 input_size,
                 hidden_size_list,
                 output_size,
                 activation='relu',
                 weight_init_std='relu',
                 weight_decay_lambda=0,
                 use_dropout=False,
                 dropout_ration=0.5,
                 use_batchnorm=False):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size_list = hidden_size_list
        self.hidden_layer_num = len(hidden_size_list)
        self.use_dropout = use_dropout
        self.weight_decay_lambda = weight_decay_lambda
        self.use_batchnorm = use_batchnorm
        self.params = {}

        # 初始化权重
        self.__init_weight(weight_init_std)

        # 生成层
        activation_layer = {'sigmoid': Sigmoid, 'relu': Relu}
        self.layers = OrderedDict()
        for idx in range(1, self.hidden_layer_num + 1):
            # print("idx:", idx)
            self.layers['Affine' + str(idx)] = Affine(
                self.params['W' + str(idx)], self.params['b' + str(idx)])
            if self.use_batchnorm:
                self.params['gamma' + str(idx)] = np.ones(
                    hidden_size_list[idx - 1])
                self.params['beta' + str(idx)] = np.zeros(
                    hidden_size_list[idx - 1])
                self.layers['BatchNorm' + str(idx)] = BatchNormalization(
                    self.params['gamma' + str(idx)],
                    self.params['beta' + str(idx)])

            self.layers['Activation_function' +
                        str(idx)] = activation_layer[activation]()

            if self.use_dropout:
                self.layers['Dropout' + str(idx)] = Dropout(dropout_ration)
        # print(self.layers)
        # print(self.params.keys())

        idx = self.hidden_layer_num + 1
        self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(idx)],
                                                  self.params['b' + str(idx)])
        # print(self.layers)
        # print(self.params.keys())

        self.last_layer = SoftmaxWithLoss()
        # print(self.last_layer)
        '''
        idx: 1
        idx: 2
        idx: 3
        idx: 4
        idx: 5
        OrderedDict(
            [
                ('Affine1', <common.layers.Affine object at 0x7fbed022e8c0>),
                ('BatchNorm1', <common.layers.BatchNormalization object at 0x7fbe9148d1e0>),
                ('Activation_function1', <common.layers.Relu object at 0x7fbe9148d120>),
                ('Dropout1', <common.layers.Dropout object at 0x7fbe9148d810>),
                ('Affine2', <common.layers.Affine object at 0x7fbe9148d990>),
                ('BatchNorm2', <common.layers.BatchNormalization object at 0x7fbe9148d9c0>),
                ('Activation_function2', <common.layers.Relu object at 0x7fbe9148d9f0>),
                ('Dropout2', <common.layers.Dropout object at 0x7fbe9148da50>),
                ('Affine3', <common.layers.Affine object at 0x7fbe9148dab0>),
                ('BatchNorm3', <common.layers.BatchNormalization object at 0x7fbe9148dae0>),
                ('Activation_function3', <common.layers.Relu object at 0x7fbe9148db10>),
                ('Dropout3', <common.layers.Dropout object at 0x7fbe9148db70>),
                ('Affine4', <common.layers.Affine object at 0x7fbe9148dbd0>),
                ('BatchNorm4', <common.layers.BatchNormalization object at 0x7fbe9148dc00>),
                ('Activation_function4', <common.layers.Relu object at 0x7fbe9148dc30>),
                ('Dropout4', <common.layers.Dropout object at 0x7fbe9148dc90>),
                ('Affine5', <common.layers.Affine object at 0x7fbe9148dcf0>),
                ('BatchNorm5', <common.layers.BatchNormalization object at 0x7fbe9148dd20>),
                ('Activation_function5', <common.layers.Relu object at 0x7fbe9148dd50>),
                ('Dropout5', <common.layers.Dropout object at 0x7fbe9148ddb0>)
            ]
        )
        dict_keys(
            ['W1', 'b1', 'W2', 'b2', 'W3', 'b3', 'W4', 'b4', 'W5', 'b5', 'W6', 'b6',
            'gamma1', 'beta1', 'gamma2', 'beta2', 'gamma3', 'beta3', 'gamma4', 'beta4', 'gamma5', 'beta5']
        )
        OrderedDict(
            [
                ('Affine1', <common.layers.Affine object at 0x7fbed022e8c0>),
                ('BatchNorm1', <common.layers.BatchNormalization object at 0x7fbe9148d1e0>),
                ('Activation_function1', <common.layers.Relu object at 0x7fbe9148d120>),
                ('Dropout1', <common.layers.Dropout object at 0x7fbe9148d810>),
                ('Affine2', <common.layers.Affine object at 0x7fbe9148d990>),
                ('BatchNorm2', <common.layers.BatchNormalization object at 0x7fbe9148d9c0>),
                ('Activation_function2', <common.layers.Relu object at 0x7fbe9148d9f0>),
                ('Dropout2', <common.layers.Dropout object at 0x7fbe9148da50>),
                ('Affine3', <common.layers.Affine object at 0x7fbe9148dab0>),
                ('BatchNorm3', <common.layers.BatchNormalization object at 0x7fbe9148dae0>),
                ('Activation_function3', <common.layers.Relu object at 0x7fbe9148db10>),
                ('Dropout3', <common.layers.Dropout object at 0x7fbe9148db70>),
                ('Affine4', <common.layers.Affine object at 0x7fbe9148dbd0>),
                ('BatchNorm4', <common.layers.BatchNormalization object at 0x7fbe9148dc00>),
                ('Activation_function4', <common.layers.Relu object at 0x7fbe9148dc30>),
                ('Dropout4', <common.layers.Dropout object at 0x7fbe9148dc90>),
                ('Affine5', <common.layers.Affine object at 0x7fbe9148dcf0>),
                ('BatchNorm5', <common.layers.BatchNormalization object at 0x7fbe9148dd20>),
                ('Activation_function5', <common.layers.Relu object at 0x7fbe9148dd50>),
                ('Dropout5', <common.layers.Dropout object at 0x7fbe9148ddb0>),
                ('Affine6', <common.layers.Affine object at 0x7fbe9148d2a0>)
            ]
        )
        dict_keys(
            ['W1', 'b1', 'W2', 'b2', 'W3', 'b3', 'W4', 'b4', 'W5', 'b5', 'W6', 'b6',
            'gamma1', 'beta1', 'gamma2', 'beta2', 'gamma3', 'beta3', 'gamma4', 'beta4', 'gamma5', 'beta5'])
        <common.layers.SoftmaxWithLoss object at 0x7fbe9148de10>
        '''

    def __init_weight(self, weight_init_std):
        """设定权重的初始值

        Parameters
        ----------
        weight_init_std : 指定权重的标准差（e.g. 0.01）
            指定'relu'或'he'的情况下设定“He的初始值”
            指定'sigmoid'或'xavier'的情况下设定“Xavier的初始值”
        """
        all_size_list = [self.input_size
                         ] + self.hidden_size_list + [self.output_size]
        # print("all_size_list:", all_size_list)

        for idx in range(1, len(all_size_list)):
            # print("idx:", idx)
            scale = weight_init_std
            if str(weight_init_std).lower() in ('relu', 'he'):
                scale = np.sqrt(2.0 /
                                all_size_list[idx - 1])  # 使用ReLU的情况下推荐的初始值
            elif str(weight_init_std).lower() in ('sigmoid', 'xavier'):
                scale = np.sqrt(1.0 /
                                all_size_list[idx - 1])  # 使用sigmoid的情况下推荐的初始值
            # print("scale:", scale)
            self.params['W' + str(idx)] = scale * np.random.randn(
                all_size_list[idx - 1], all_size_list[idx])
            self.params['b' + str(idx)] = np.zeros(all_size_list[idx])
            # print('W' + str(idx))
            # print('b' + str(idx))
        '''
        all_size_list: [784, 100, 100, 100, 100, 100, 10]
        idx: 1
        scale: 0.01
        W1
        b1
        idx: 2
        scale: 0.01
        W2
        b2
        idx: 3
        scale: 0.01
        W3
        b3
        idx: 4
        scale: 0.01
        W4
        b4
        idx: 5
        scale: 0.01
        W5
        b5
        idx: 6
        scale: 0.01
        W6
        b6
        '''

    def predict(self, x, train_flg=False):
        for key, layer in self.layers.items():
            if "Dropout" in key or "BatchNorm" in key:
                x = layer.forward(x, train_flg)
            else:
                x = layer.forward(x)

        return x

    def loss(self, x, t, train_flg=False):
        """求损失函数
        参数x是输入数据，t是教师标签
        """
        y = self.predict(x, train_flg)

        weight_decay = 0
        for idx in range(1, self.hidden_layer_num + 2):
            W = self.params['W' + str(idx)]
            weight_decay += 0.5 * self.weight_decay_lambda * np.sum(W**2)

        return self.last_layer.forward(y, t) + weight_decay

    def accuracy(self, X, T):
        Y = self.predict(X, train_flg=False)
        Y = np.argmax(Y, axis=1)
        if T.ndim != 1: T = np.argmax(T, axis=1)

        accuracy = np.sum(Y == T) / float(X.shape[0])
        return accuracy

    def numerical_gradient(self, X, T):
        """求梯度（数值微分）

        Parameters
        ----------
        X : 输入数据
        T : 教师标签

        Returns
        -------
        具有各层的梯度的字典变量
            grads['W1']、grads['W2']、...是各层的权重
            grads['b1']、grads['b2']、...是各层的偏置
        """
        loss_W = lambda W: self.loss(X, T, train_flg=True)

        grads = {}
        for idx in range(1, self.hidden_layer_num + 2):
            grads['W' + str(idx)] = numerical_gradient(
                loss_W, self.params['W' + str(idx)])
            grads['b' + str(idx)] = numerical_gradient(
                loss_W, self.params['b' + str(idx)])

            if self.use_batchnorm and idx != self.hidden_layer_num + 1:
                grads['gamma' + str(idx)] = numerical_gradient(
                    loss_W, self.params['gamma' + str(idx)])
                grads['beta' + str(idx)] = numerical_gradient(
                    loss_W, self.params['beta' + str(idx)])

        return grads

    def gradient(self, x, t):
        # forward
        self.loss(x, t, train_flg=True)

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
                idx)].dW + self.weight_decay_lambda * self.params['W' +
                                                                  str(idx)]
            grads['b' + str(idx)] = self.layers['Affine' + str(idx)].db

            if self.use_batchnorm and idx != self.hidden_layer_num + 1:
                grads['gamma' + str(idx)] = self.layers['BatchNorm' +
                                                        str(idx)].dgamma
                grads['beta' + str(idx)] = self.layers['BatchNorm' +
                                                       str(idx)].dbeta

        return grads


if __name__ == "__main__":
    bn_network = MultiLayerNetExtend(
        input_size=784,
        hidden_size_list=[100, 100, 100, 100, 100],
        output_size=10,
        weight_init_std=0.01,
        use_batchnorm=True,
        use_dropout=True)
    print(bn_network.input_size)  # 784
    print(bn_network.output_size)  # 10
    print(bn_network.hidden_size_list)  # [100, 100, 100, 100, 100]
    print(bn_network.hidden_layer_num)  # 5
    print(bn_network.use_dropout)  # True
    print(bn_network.weight_decay_lambda)  # 0
    print(bn_network.use_batchnorm)  # True
    print(bn_network.params["W1"].shape)  # (784, 100)
    print(bn_network.params["b1"].shape)  # (100,)
    print(bn_network.params["W2"].shape)  # (100, 100)
    print(bn_network.params["b2"].shape)  # (100,)
    print(bn_network.params["W3"].shape)  # (100, 100)
    print(bn_network.params["b3"].shape)  # (100,)
    print(bn_network.params["W4"].shape)  # (100, 100)
    print(bn_network.params["b4"].shape)  # (100,)
    print(bn_network.params["W5"].shape)  # (100, 100)
    print(bn_network.params["b5"].shape)  # (100,)
    print(bn_network.params["W6"].shape)  # (100, 10)
    print(bn_network.params["b6"].shape)  # (10,)
