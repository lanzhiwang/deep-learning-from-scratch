# coding: utf-8
import numpy as np
import matplotlib.pylab as plt
from matplotlib import pyplot
"""
>>> import numpy as np
>>> X = np.arange(-5.0, 5.0, 0.1)
>>> np.exp(-X)
array([1.48413159e+02, 1.34289780e+02, 1.21510418e+02, 1.09947172e+02,
       9.94843156e+01, 9.00171313e+01, 8.14508687e+01, 7.36997937e+01,
       6.66863310e+01, 6.03402876e+01, 5.45981500e+01, 4.94024491e+01,
       4.47011845e+01, 4.04473044e+01, 3.65982344e+01, 3.31154520e+01,
       2.99641000e+01, 2.71126389e+01, 2.45325302e+01, 2.21979513e+01,
       2.00855369e+01, 1.81741454e+01, 1.64446468e+01, 1.48797317e+01,
       1.34637380e+01, 1.21824940e+01, 1.10231764e+01, 9.97418245e+00,
       9.02501350e+00, 8.16616991e+00, 7.38905610e+00, 6.68589444e+00,
       6.04964746e+00, 5.47394739e+00, 4.95303242e+00, 4.48168907e+00,
       4.05519997e+00, 3.66929667e+00, 3.32011692e+00, 3.00416602e+00,
       2.71828183e+00, 2.45960311e+00, 2.22554093e+00, 2.01375271e+00,
       1.82211880e+00, 1.64872127e+00, 1.49182470e+00, 1.34985881e+00,
       1.22140276e+00, 1.10517092e+00, 1.00000000e+00, 9.04837418e-01,
       8.18730753e-01, 7.40818221e-01, 6.70320046e-01, 6.06530660e-01,
       5.48811636e-01, 4.96585304e-01, 4.49328964e-01, 4.06569660e-01,
       3.67879441e-01, 3.32871084e-01, 3.01194212e-01, 2.72531793e-01,
       2.46596964e-01, 2.23130160e-01, 2.01896518e-01, 1.82683524e-01,
       1.65298888e-01, 1.49568619e-01, 1.35335283e-01, 1.22456428e-01,
       1.10803158e-01, 1.00258844e-01, 9.07179533e-02, 8.20849986e-02,
       7.42735782e-02, 6.72055127e-02, 6.08100626e-02, 5.50232201e-02,
       4.97870684e-02, 4.50492024e-02, 4.07622040e-02, 3.68831674e-02,
       3.33732700e-02, 3.01973834e-02, 2.73237224e-02, 2.47235265e-02,
       2.23707719e-02, 2.02419114e-02, 1.83156389e-02, 1.65726754e-02,
       1.49955768e-02, 1.35685590e-02, 1.22773399e-02, 1.11089965e-02,
       1.00518357e-02, 9.09527710e-03, 8.22974705e-03, 7.44658307e-03])
>>>
"""


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


X = np.arange(-5.0, 5.0, 0.1)
Y = sigmoid(X)
plt.plot(X, Y)
plt.ylim(-0.1, 1.1)
plt.show()
pyplot.savefig('sigmoid.png')
