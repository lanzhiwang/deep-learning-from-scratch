# coding: utf-8
import numpy as np
import matplotlib.pylab as plt
from matplotlib import pyplot


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def step_function(x):
    return np.array(x > 0, dtype=np.int32)


x = np.arange(-5.0, 5.0, 0.1)
y1 = sigmoid(x)
y2 = step_function(x)

plt.plot(x, y1)
plt.plot(x, y2, 'k--')
plt.ylim(-0.1, 1.1)  #指定图中绘制的y轴的范围
plt.show()
pyplot.savefig('sig_step_compare.png')
