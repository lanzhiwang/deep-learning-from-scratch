# coding: utf-8

import numpy as np


def softmax1(a):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y


# >>> import numpy as np
# >>> a = np.array([0.3, 2.9, 4.0])
# >>> exp_a = np.exp(a)
# >>> exp_a
# array([ 1.34985881, 18.17414537, 54.59815003])
# >>> sum_exp_a = np.sum(exp_a)
# >>> sum_exp_a
# 74.1221542101633
# >>> y = exp_a / sum_exp_a
# >>> y
# array([0.01821127, 0.24519181, 0.73659691])
# >>>


def softmax2(a):
    c = np.max(a)
    exp_a = np.exp(a - c)  # 溢出对策
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y


# >>> import numpy as np
# >>> a = np.array([1010, 1000, 990])
# >>> np.exp(a) / np.sum(np.exp(a))
# <stdin>:1: RuntimeWarning: overflow encountered in exp
# <stdin>:1: RuntimeWarning: invalid value encountered in divide
# array([nan, nan, nan])
# >>> c = np.max(a)
# >>> a - c
# array([  0, -10, -20])
# >>> np.exp(a - c) / np.sum(np.exp(a - c))
# array([9.99954600e-01, 4.53978686e-05, 2.06106005e-09])
# >>>
