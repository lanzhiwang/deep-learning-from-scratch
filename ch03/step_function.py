# coding: utf-8
import numpy as np
import matplotlib.pylab as plt


def step_function(x):
    return np.array(x > 0, dtype=np.int)


# >>> import numpy as np
# >>> x = np.array([-1.0, 1.0, 2.0])
# >>> x
# array([-1.,  1.,  2.])
# >>> y = x > 0
# >>> y
# array([False,  True,  True])
# >>> y = y.astype(np.int)
# >>> y
# array([0, 1, 1])
# >>>

# >>> import numpy as np
# >>> x = np.array([-1.0, 1.0, 2.0])
# >>> y = np.array(x > 0, dtype=np.int)
# >>> y
# array([0, 1, 1])
# >>>

X = np.arange(-5.0, 5.0, 0.1)
Y = step_function(X)
plt.plot(X, Y)
plt.ylim(-0.1, 1.1)  # 指定图中绘制的y轴的范围
plt.show()
