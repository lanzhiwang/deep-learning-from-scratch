# coding: utf-8
import sys, os

sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
from common.functions import *
from common.gradient import numerical_gradient


class TwoLayerNet:

    # TwoLayerNet(input_size=6, hidden_size=4, output_size=3)
    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size,
                 weight_init_std=0.01):
        # 初始化权重
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(
            input_size, hidden_size)  # 6, 4
        self.params['b1'] = np.zeros(hidden_size)  # 4
        self.params['W2'] = weight_init_std * np.random.randn(
            hidden_size, output_size)  # 4, 3
        self.params['b2'] = np.zeros(output_size)  # 3

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        return y

    # x:输入数据, t:监督数据
    def loss(self, x, t):
        y = self.predict(x)

        return cross_entropy_error(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
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
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        grads = {}

        batch_num = x.shape[0]

        # forward
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        # backward
        dy = (y - t) / batch_num
        grads['W2'] = np.dot(z1.T, dy)
        grads['b2'] = np.sum(dy, axis=0)

        da1 = np.dot(dy, W2.T)
        dz1 = sigmoid_grad(a1) * da1
        grads['W1'] = np.dot(x.T, dz1)
        grads['b1'] = np.sum(dz1, axis=0)

        return grads


'''
net = TwoLayerNet(input_size=6, hidden_size=4, output_size=3)
print(net.params['W1'])
# [[-0.00978732 -0.00320251  0.01526357 -0.02275537]
#  [ 0.00032345 -0.00540361 -0.01313697  0.00033448]
#  [-0.00648541 -0.0102803   0.00359513  0.00671428]
#  [-0.01576736  0.00866359  0.00289414  0.00733741]
#  [-0.00220846 -0.01015147 -0.00339344 -0.00419903]
#  [-0.00628765  0.01626594  0.00392469 -0.00228071]]
print(net.params['W1'].shape)  # (6, 4)
print(net.params['b1'])
# [0. 0. 0. 0.]
print(net.params['b1'].shape)  # (4,)
print(net.params['W2'])
# [[ 0.00192098  0.00024825  0.01340817]
#  [ 0.01243373 -0.02129871 -0.0174414 ]
#  [-0.00271292 -0.00575692 -0.0075212 ]
#  [-0.00588021 -0.00284449  0.0097602 ]]

print(net.params['W2'].shape)  # (4, 3)
print(net.params['b2'])
# [0. 0. 0.]
print(net.params['b2'].shape)  # (3,)

x = np.random.rand(5, 6)
print("x:", x)
# x: [[0.75910211 0.65368807 0.758838   0.11667926 0.60549841 0.31275868]
#  [0.32730519 0.48436223 0.69891347 0.62865253 0.58399253 0.15270768]
#  [0.07350335 0.40038194 0.5914738  0.49844018 0.3333124  0.44796918]
#  [0.56689846 0.71581193 0.97935238 0.53009585 0.87217687 0.02245043]
#  [0.00483501 0.21798065 0.2974696  0.4459156  0.38443084 0.79673158]]

y = net.predict(x)
print("y:", y)
# y: [[0.33570831 0.32985223 0.33443945]
#  [0.33570989 0.32984463 0.33444549]
#  [0.33572137 0.32983297 0.33444565]
#  [0.33569405 0.32985818 0.33444777]
#  [0.33573818 0.32982371 0.33443811]]

t = np.random.rand(5, 3)
print("t:", t)
# t: [[0.97514354 0.85031743 0.98255742]
#  [0.48349918 0.10980192 0.07659049]
#  [0.36065803 0.07684544 0.48442066]
#  [0.82122246 0.00915638 0.98770195]
#  [0.82850648 0.45565583 0.88320192]]

grads = net.numerical_gradient(x, t)
print(grads['W1'])
# [[ 2.05264821e-04 -4.08436256e-04 -3.13661347e-04  2.90696813e-04]
#  [ 1.54737141e-04 -5.79710993e-04 -6.71284095e-05  7.16398285e-04]
#  [ 7.75716602e-05 -5.80271076e-04  1.46780519e-04  8.88983064e-04]
#  [-7.78080078e-05 -7.09922431e-04  6.57572605e-04  1.47168089e-03]
#  [-7.33782457e-05 -7.39064540e-04  6.63467037e-04  1.51518783e-03]
#  [ 2.59305688e-05 -6.49114654e-04  3.32586303e-04  1.12975657e-03]]

print(grads['W1'].shape)  # (6, 4)

print(grads['b1'])
# [ 0.00010799 -0.00129326  0.00050675  0.00212567]

print(grads['b1'].shape)  # (4,)
print(grads['W2'])
# [[-0.23369424  0.06818808  0.16550616]
#  [-0.23581962  0.06911994  0.16669969]
#  [-0.23385005  0.06877946  0.16507059]
#  [-0.23622128  0.0690939   0.16712737]]

print(grads['W2'].shape)  # (4, 3)

print(grads['b2'])
# [-0.46875653  0.13694212  0.3318144 ]
print(grads['b2'].shape)  # (3,)
'''
