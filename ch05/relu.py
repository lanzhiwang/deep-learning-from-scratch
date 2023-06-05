# coding: utf-8
import sys, os

sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import numpy as np
from common.layers import *

x = np.array([[1.0, -0.5], [-2.0, 3.0]])
print(x)
relu = Relu()
print(relu.forward(x))
# [[ 1.  -0.5]
#  [-2.   3. ]]
# mask: [[False  True]
#  [ True False]]
# [[1. 0.]
#  [0. 3.]]

print(relu.backward(np.array([[2.0, -2.5], [-4.0, 6.0]])))
# [[2. 0.]
#  [0. 6.]]
