"""
>>> import numpy as np
>>> def function_2(x):
...     return np.sum(x**2)
...
>>> x1 = np.array([1, 2, 3, 4])
>>> x1
array([1, 2, 3, 4])
>>> x1**2
array([ 1,  4,  9, 16])
>>> np.sum(x1**2)
np.int64(30)
>>>
>>> x2 = np.array([[1, 2, 3, 4], [4, 3, 2, 1]])
>>> x2
array([[1, 2, 3, 4],
       [4, 3, 2, 1]])
>>> x2**2
array([[ 1,  4,  9, 16],
       [16,  9,  4,  1]])
>>> np.sum(x2**2)
np.int64(60)
"""
