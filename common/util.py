# coding: utf-8
import numpy as np


def smooth_curve(x):
    """用于使损失函数的图形变圆滑

    参考：http://glowingpython.blogspot.jp/2012/02/convolution-with-numpy.html
    """
    window_len = 11
    s = np.r_[x[window_len - 1:0:-1], x, x[-1:-window_len:-1]]
    w = np.kaiser(window_len, 2)
    y = np.convolve(w / w.sum(), s, mode='valid')
    return y[5:len(y) - 5]


def shuffle_dataset(x, t):
    """打乱数据集

    Parameters
    ----------
    x : 训练数据
    t : 监督数据

    Returns
    -------
    x, t : 打乱的训练数据和监督数据
    """
    permutation = np.random.permutation(x.shape[0])
    x = x[permutation, :] if x.ndim == 2 else x[permutation, :, :, :]
    t = t[permutation]

    return x, t


def conv_output_size(input_size, filter_size, stride=1, pad=0):
    return (input_size + 2 * pad - filter_size) / stride + 1


def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """
    Parameters
    ----------
    input_data: 由(数据量, 通道, 高, 长)的 4 维数组构成的输入数据
    filter_h: 滤波器的高
    filter_w: 滤波器的长
    stride: 步幅
    pad: 填充

    Returns
    -------
    col: 2 维数组

    >>> import sys, os
    >>> sys.path.append(os.pardir)
    >>> from common.util import im2col
    >>> import numpy as np
    >>> A = np.array([1, 2, 3, 0, 0, 1, 2, 3, 3, 0, 1, 2, 2, 3, 0, 1]).reshape(1, 1, 4, 4)
    >>> A
    array([[[[1, 2, 3, 0],
             [0, 1, 2, 3],
             [3, 0, 1, 2],
             [2, 3, 0, 1]]]])
    >>> a_col = im2col(A, 3, 3, stride=1, pad=0)
    N, C, H, W: 1 1 4 4
    out_h: 2
    out_w: 2
    img: (1, 1, 4, 4)
    col: (1, 1, 3, 3, 2, 2)
    >>> a_col.shape
    (4, 9)
    >>> a_col
    array([[1., 2., 3., 0., 1., 2., 3., 0., 1.],
           [2., 3., 0., 1., 2., 3., 0., 1., 2.],
           [0., 1., 2., 3., 0., 1., 2., 3., 0.],
           [1., 2., 3., 0., 1., 2., 3., 0., 1.]])
    >>>
    >>> A = np.array([1, 2, 3, 0, 0, 1, 2, 3, 3, 0, 1, 2, 2, 3, 0, 1, 1, 2, 3, 0, 0, 1, 2, 3, 3, 0, 1, 2, 2, 3, 0, 1]).reshape(2, 1, 4, 4)
    >>> A
    array([[[[1, 2, 3, 0],
             [0, 1, 2, 3],
             [3, 0, 1, 2],
             [2, 3, 0, 1]]],


           [[[1, 2, 3, 0],
             [0, 1, 2, 3],
             [3, 0, 1, 2],
             [2, 3, 0, 1]]]])
    >>> a_col = im2col(A, 3, 3, stride=1, pad=0)
    N, C, H, W: 2 1 4 4
    out_h: 2
    out_w: 2
    img: (2, 1, 4, 4)
    col: (2, 1, 3, 3, 2, 2)
    >>> a_col.shape
    (8, 9)
    >>> a_col
    array([[1., 2., 3., 0., 1., 2., 3., 0., 1.],
           [2., 3., 0., 1., 2., 3., 0., 1., 2.],
           [0., 1., 2., 3., 0., 1., 2., 3., 0.],
           [1., 2., 3., 0., 1., 2., 3., 0., 1.],
           [1., 2., 3., 0., 1., 2., 3., 0., 1.],
           [2., 3., 0., 1., 2., 3., 0., 1., 2.],
           [0., 1., 2., 3., 0., 1., 2., 3., 0.],
           [1., 2., 3., 0., 1., 2., 3., 0., 1.]])
    >>>
    """
    N, C, H, W = input_data.shape
    # print("N, C, H, W:", N, C, H, W)  # N, C, H, W: 10 3 7 7

    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1
    # print("out_h:", out_h)  # out_h: 5
    # print("out_w:", out_w)  # out_w: 5

    img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)],
                 'constant')
    # print("img:", img.shape)# img: (10, 3, 9, 9)

    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))
    # print("col:", col.shape)  # col: (10, 3, 5, 5, 5, 5)

    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    return col


def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    """

    Parameters
    ----------
    col:
    input_shape: 输入数据的形状(例: (10, 1, 28, 28))
    filter_h:
    filter_w
    stride
    pad

    Returns
    -------
    """
    N, C, H, W = input_shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h,
                      filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2 * pad + stride - 1, W + 2 * pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]
