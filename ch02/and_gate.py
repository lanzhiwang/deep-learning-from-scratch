# coding: utf-8
import numpy as np


def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


# >>> import numpy as np
# >>> x = np.array([0, 1])
# >>> w = np.array([0.5, 0.5])
# >>> b = -0.7
# >>> w * x
# array([0. , 0.5])
# >>> np.sum(w * x)
# 0.5
# >>> np.sum(w * x) + b
# -0.19999999999999996
# >>>

if __name__ == '__main__':
    for xs in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        y = AND(xs[0], xs[1])
        print(str(xs) + " -> " + str(y))
