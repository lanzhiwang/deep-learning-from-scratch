# coding: utf-8
try:
    import urllib.request
except ImportError:
    raise ImportError('You should use Python 3.x')
import os.path
import gzip
import pickle
import os
import numpy as np

url_base = 'http://yann.lecun.com/exdb/mnist/'
key_file = {
    'train_img': 'train-images-idx3-ubyte.gz',
    'train_label': 'train-labels-idx1-ubyte.gz',
    'test_img': 't10k-images-idx3-ubyte.gz',
    'test_label': 't10k-labels-idx1-ubyte.gz'
}

dataset_dir = os.path.dirname(os.path.abspath(__file__))
# print(__file__)  # /deep-learning-from-scratch/dataset/mnist.py
# print(
#     os.path.abspath(__file__))  # /deep-learning-from-scratch/dataset/mnist.py
# print(dataset_dir)  # /deep-learning-from-scratch/dataset

save_file = dataset_dir + "/mnist.pkl"
# print(save_file)  # /deep-learning-from-scratch/dataset/mnist.pkl

train_num = 60000
test_num = 10000
img_dim = (1, 28, 28)
img_size = 784


def _download(file_name):
    # print("_download file_name:", file_name)
    # _download file_name: train-images-idx3-ubyte.gz
    # _download file_name: train-labels-idx1-ubyte.gz
    # _download file_name: t10k-images-idx3-ubyte.gz
    # _download file_name: t10k-labels-idx1-ubyte.gz

    file_path = dataset_dir + "/" + file_name
    # print("_download file_path:", file_path)
    # _download file_path: /deep-learning-from-scratch/dataset/train-images-idx3-ubyte.gz
    # _download file_path: /deep-learning-from-scratch/dataset/train-labels-idx1-ubyte.gz
    # _download file_path: /deep-learning-from-scratch/dataset/t10k-images-idx3-ubyte.gz
    # _download file_path: /deep-learning-from-scratch/dataset/t10k-labels-idx1-ubyte.gz

    if os.path.exists(file_path):
        return

    print("Downloading " + file_name + " ... ")
    # print("_download url:", url_base + file_name, file_path)
    # _download url: http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz /deep-learning-from-scratch/dataset/train-images-idx3-ubyte.gz
    # _download url: http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz /deep-learning-from-scratch/dataset/train-labels-idx1-ubyte.gz
    # _download url: http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz /deep-learning-from-scratch/dataset/t10k-images-idx3-ubyte.gz
    # _download url: http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz /deep-learning-from-scratch/dataset/t10k-labels-idx1-ubyte.gz

    urllib.request.urlretrieve(url_base + file_name, file_path)
    print("Done")


def download_mnist():
    for v in key_file.values():
        _download(v)


def _load_label(file_name):
    file_path = dataset_dir + "/" + file_name
    # print("_load_label file_path:", file_path)
    # _load_label file_path: /deep-learning-from-scratch/dataset/train-labels-idx1-ubyte.gz

    print("Converting " + file_name + " to NumPy Array ...")
    with gzip.open(file_path, 'rb') as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8)
        # print("_load_label labels:", labels)
        # _load_label labels: [5 0 4 ... 5 6 8]
    print("Done")

    return labels


def _load_img(file_name):
    file_path = dataset_dir + "/" + file_name
    # print("_load_img file_path:", file_path)
    # _load_img file_path: /deep-learning-from-scratch/dataset/train-images-idx3-ubyte.gz

    print("Converting " + file_name + " to NumPy Array ...")
    with gzip.open(file_path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
        # print("_load_img data:", data)
        # _load_img data: [0 0 0 ... 0 0 0]
    data = data.reshape(-1, img_size)
    # print("_load_img data:", data)
    # _load_img data: [[0 0 0 ... 0 0 0]
    #                  [0 0 0 ... 0 0 0]
    #                  [0 0 0 ... 0 0 0]
    #                  ...
    #                  [0 0 0 ... 0 0 0]
    #                  [0 0 0 ... 0 0 0]
    #                  [0 0 0 ... 0 0 0]]

    print("Done")

    return data


def _convert_numpy():
    dataset = {}
    dataset['train_img'] = _load_img(key_file['train_img'])
    dataset['train_label'] = _load_label(key_file['train_label'])
    dataset['test_img'] = _load_img(key_file['test_img'])
    dataset['test_label'] = _load_label(key_file['test_label'])

    return dataset


def init_mnist():
    download_mnist()
    dataset = _convert_numpy()
    # print("dataset:", dataset)
    # dataset:
    # {
    #     'train_img': array(
    #         [
    #             [0, 0, 0, ..., 0, 0, 0],
    #             [0, 0, 0, ..., 0, 0, 0],
    #             [0, 0, 0, ..., 0, 0, 0],
    #             ...,
    #             [0, 0, 0, ..., 0, 0, 0],
    #             [0, 0, 0, ..., 0, 0, 0],
    #             [0, 0, 0, ..., 0, 0, 0]
    #         ], dtype=uint8
    #     ),
    #     'train_label': array([5, 0, 4, ..., 5, 6, 8], dtype=uint8),
    #     'test_img': array(
    #         [
    #             [0, 0, 0, ..., 0, 0, 0],
    #             [0, 0, 0, ..., 0, 0, 0],
    #             [0, 0, 0, ..., 0, 0, 0],
    #             ...,
    #             [0, 0, 0, ..., 0, 0, 0],
    #             [0, 0, 0, ..., 0, 0, 0],
    #             [0, 0, 0, ..., 0, 0, 0]
    #         ], dtype=uint8
    #     ),
    #     'test_label': array([7, 2, 1, ..., 4, 5, 6], dtype=uint8)
    # }
    print("Creating pickle file ...")
    with open(save_file, 'wb') as f:
        pickle.dump(dataset, f, -1)
    print("Done!")


def _change_one_hot_label(X):
    # print("_change_one_hot_label X:", X)
    # _change_one_hot_label X: [5 0 4 ... 5 6 8]

    # print("_change_one_hot_label X:", X.shape)
    # _change_one_hot_label X: (60000,)

    T = np.zeros((X.size, 10))

    # print("_change_one_hot_label T:", T)
    # _change_one_hot_label T: [[0. 0. 0. ... 0. 0. 0.]
    #                         [0. 0. 0. ... 0. 0. 0.]
    #                         [0. 0. 0. ... 0. 0. 0.]
    #                         ...
    #                         [0. 0. 0. ... 0. 0. 0.]
    #                         [0. 0. 0. ... 0. 0. 0.]
    #                         [0. 0. 0. ... 0. 0. 0.]]

    # print("_change_one_hot_label T:", T.shape)
    # _change_one_hot_label T: (60000, 10)

    for idx, row in enumerate(T):
        row[X[idx]] = 1

    return T


# >>> import numpy as np
# >>> X = np.array([3, 7, 2, 5, 8, 4, 1, 7, 9])
# >>> X
# array([3, 7, 2, 5, 8, 4, 1, 7, 9])
# >>> X.shape
# (9,)
# >>> X.size
# 9
# >>> T = np.zeros((X.size, 10))
# >>> T
# array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
# >>> T.shape
# (9, 10)
# >>>
# >>> for idx, row in enumerate(T):
# ...     print(idx, row)
# ...
# 0 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
# 1 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
# 2 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
# 3 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
# 4 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
# 5 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
# 6 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
# 7 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
# 8 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
# >>> for idx, row in enumerate(T):
# ...     print(idx, row)
# ...     print(X[idx])
# ...
# 0 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
# 3
# 1 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
# 7
# 2 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
# 2
# 3 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
# 5
# 4 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
# 8
# 5 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
# 4
# 6 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
# 1
# 7 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
# 7
# 8 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
# 9
# >>>
# >>> for idx, row in enumerate(T):
# ...     row[X[idx]] = 1

# >>> T
# array([[0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
#        [0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
#        [0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
#        [0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
#        [0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
#        [0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
#        [0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
#        [0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
#        [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]])
# >>>


def load_mnist(normalize=True, flatten=True, one_hot_label=False):
    """读入MNIST数据集

    Parameters
    ----------
    normalize : 将图像的像素值正规化为0.0~1.0
    one_hot_label :
        one_hot_label为True的情况下，标签作为one-hot数组返回
        one-hot数组是指[0,0,1,0,0,0,0,0,0,0]这样的数组
    flatten : 是否将图像展开为一维数组

    Returns
    -------
    (训练图像, 训练标签), (测试图像, 测试标签)
    """
    if not os.path.exists(save_file):
        init_mnist()

    with open(save_file, 'rb') as f:
        dataset = pickle.load(f)

    if normalize:
        # 第 1 个 参 数 normalize 设置是否将输入图像正规化为 0.0 ~ 1.0 的值。
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].astype(np.float32)
            dataset[key] /= 255.0

    if one_hot_label:
        # 第 3 个参数 one_hot_label 设置是否将标签保存为 one-hot 表示(one-hot representation)
        dataset['train_label'] = _change_one_hot_label(dataset['train_label'])
        dataset['test_label'] = _change_one_hot_label(dataset['test_label'])

    if not flatten:
        # 第 2 个参数 flatten 设置是否展开输入图像 (变成一维数组)。
        # 如果将该参数设置为 False，则输入图像为 1 × 28 × 28 的三维数组;
        # 若设置为 True，则输入图像会保存为由 784 个 元素构成的一维数组。
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].reshape(-1, 1, 28, 28)

    return (dataset['train_img'],
            dataset['train_label']), (dataset['test_img'],
                                      dataset['test_label'])


if __name__ == '__main__':
    init_mnist()
    print("*****************************************************************")

    (x_train, t_train), (x_test, t_test) = load_mnist()
    print("x_train:", x_train)
    print(x_train.shape)
    print("t_train:", t_train)
    print(t_train.shape)
    print("x_test:", x_test)
    print(x_test.shape)
    print("t_test:", t_test)
    print(x_test.shape)
    print("*****************************************************************")

    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=False)
    print("x_train:", x_train)
    print(x_train.shape)
    print("t_train:", t_train)
    print(t_train.shape)
    print("x_test:", x_test)
    print(x_test.shape)
    print("t_test:", t_test)
    print(x_test.shape)
    print("*****************************************************************")

    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)
    print("x_train:", x_train)
    print(x_train.shape)
    print("t_train:", t_train)
    print(t_train.shape)
    print("x_test:", x_test)
    print(x_test.shape)
    print("t_test:", t_test)
    print(x_test.shape)
    print("*****************************************************************")

    (x_train, t_train), (x_test, t_test) = load_mnist(one_hot_label=True)
    print("x_train:", x_train)
    print(x_train.shape)
    print("t_train:", t_train)
    print(t_train.shape)
    print("x_test:", x_test)
    print(x_test.shape)
    print("t_test:", t_test)
    print(x_test.shape)
    print("*****************************************************************")

# $ python mnist.py
# Converting train-images-idx3-ubyte.gz to NumPy Array ...
# Done
# Converting train-labels-idx1-ubyte.gz to NumPy Array ...
# Done
# Converting t10k-images-idx3-ubyte.gz to NumPy Array ...
# Done
# Converting t10k-labels-idx1-ubyte.gz to NumPy Array ...
# Done
# Creating pickle file ...
# Done!
# *****************************************************************
# x_train: [[0. 0. 0. ... 0. 0. 0.]
#  [0. 0. 0. ... 0. 0. 0.]
#  [0. 0. 0. ... 0. 0. 0.]
#  ...
#  [0. 0. 0. ... 0. 0. 0.]
#  [0. 0. 0. ... 0. 0. 0.]
#  [0. 0. 0. ... 0. 0. 0.]]
# (60000, 784)
# t_train: [5 0 4 ... 5 6 8]
# (60000,)
# x_test: [[0. 0. 0. ... 0. 0. 0.]
#  [0. 0. 0. ... 0. 0. 0.]
#  [0. 0. 0. ... 0. 0. 0.]
#  ...
#  [0. 0. 0. ... 0. 0. 0.]
#  [0. 0. 0. ... 0. 0. 0.]
#  [0. 0. 0. ... 0. 0. 0.]]
# (10000, 784)
# t_test: [7 2 1 ... 4 5 6]
# (10000, 784)
# *****************************************************************
# x_train: [[0 0 0 ... 0 0 0]
#  [0 0 0 ... 0 0 0]
#  [0 0 0 ... 0 0 0]
#  ...
#  [0 0 0 ... 0 0 0]
#  [0 0 0 ... 0 0 0]
#  [0 0 0 ... 0 0 0]]
# (60000, 784)
# t_train: [5 0 4 ... 5 6 8]
# (60000,)
# x_test: [[0 0 0 ... 0 0 0]
#  [0 0 0 ... 0 0 0]
#  [0 0 0 ... 0 0 0]
#  ...
#  [0 0 0 ... 0 0 0]
#  [0 0 0 ... 0 0 0]
#  [0 0 0 ... 0 0 0]]
# (10000, 784)
# t_test: [7 2 1 ... 4 5 6]
# (10000, 784)
# *****************************************************************
# x_train: [[[[0. 0. 0. ... 0. 0. 0.]
#    [0. 0. 0. ... 0. 0. 0.]
#    [0. 0. 0. ... 0. 0. 0.]
#    ...
#    [0. 0. 0. ... 0. 0. 0.]
#    [0. 0. 0. ... 0. 0. 0.]
#    [0. 0. 0. ... 0. 0. 0.]]]

#  [[[0. 0. 0. ... 0. 0. 0.]
#    [0. 0. 0. ... 0. 0. 0.]
#    [0. 0. 0. ... 0. 0. 0.]
#    ...
#    [0. 0. 0. ... 0. 0. 0.]
#    [0. 0. 0. ... 0. 0. 0.]
#    [0. 0. 0. ... 0. 0. 0.]]]

#  [[[0. 0. 0. ... 0. 0. 0.]
#    [0. 0. 0. ... 0. 0. 0.]
#    [0. 0. 0. ... 0. 0. 0.]
#    ...
#    [0. 0. 0. ... 0. 0. 0.]
#    [0. 0. 0. ... 0. 0. 0.]
#    [0. 0. 0. ... 0. 0. 0.]]]

#  ...

#  [[[0. 0. 0. ... 0. 0. 0.]
#    [0. 0. 0. ... 0. 0. 0.]
#    [0. 0. 0. ... 0. 0. 0.]
#    ...
#    [0. 0. 0. ... 0. 0. 0.]
#    [0. 0. 0. ... 0. 0. 0.]
#    [0. 0. 0. ... 0. 0. 0.]]]

#  [[[0. 0. 0. ... 0. 0. 0.]
#    [0. 0. 0. ... 0. 0. 0.]
#    [0. 0. 0. ... 0. 0. 0.]
#    ...
#    [0. 0. 0. ... 0. 0. 0.]
#    [0. 0. 0. ... 0. 0. 0.]
#    [0. 0. 0. ... 0. 0. 0.]]]

#  [[[0. 0. 0. ... 0. 0. 0.]
#    [0. 0. 0. ... 0. 0. 0.]
#    [0. 0. 0. ... 0. 0. 0.]
#    ...
#    [0. 0. 0. ... 0. 0. 0.]
#    [0. 0. 0. ... 0. 0. 0.]
#    [0. 0. 0. ... 0. 0. 0.]]]]
# (60000, 1, 28, 28)
# t_train: [5 0 4 ... 5 6 8]
# (60000,)
# x_test: [[[[0. 0. 0. ... 0. 0. 0.]
#    [0. 0. 0. ... 0. 0. 0.]
#    [0. 0. 0. ... 0. 0. 0.]
#    ...
#    [0. 0. 0. ... 0. 0. 0.]
#    [0. 0. 0. ... 0. 0. 0.]
#    [0. 0. 0. ... 0. 0. 0.]]]

#  [[[0. 0. 0. ... 0. 0. 0.]
#    [0. 0. 0. ... 0. 0. 0.]
#    [0. 0. 0. ... 0. 0. 0.]
#    ...
#    [0. 0. 0. ... 0. 0. 0.]
#    [0. 0. 0. ... 0. 0. 0.]
#    [0. 0. 0. ... 0. 0. 0.]]]

#  [[[0. 0. 0. ... 0. 0. 0.]
#    [0. 0. 0. ... 0. 0. 0.]
#    [0. 0. 0. ... 0. 0. 0.]
#    ...
#    [0. 0. 0. ... 0. 0. 0.]
#    [0. 0. 0. ... 0. 0. 0.]
#    [0. 0. 0. ... 0. 0. 0.]]]

#  ...

#  [[[0. 0. 0. ... 0. 0. 0.]
#    [0. 0. 0. ... 0. 0. 0.]
#    [0. 0. 0. ... 0. 0. 0.]
#    ...
#    [0. 0. 0. ... 0. 0. 0.]
#    [0. 0. 0. ... 0. 0. 0.]
#    [0. 0. 0. ... 0. 0. 0.]]]

#  [[[0. 0. 0. ... 0. 0. 0.]
#    [0. 0. 0. ... 0. 0. 0.]
#    [0. 0. 0. ... 0. 0. 0.]
#    ...
#    [0. 0. 0. ... 0. 0. 0.]
#    [0. 0. 0. ... 0. 0. 0.]
#    [0. 0. 0. ... 0. 0. 0.]]]

#  [[[0. 0. 0. ... 0. 0. 0.]
#    [0. 0. 0. ... 0. 0. 0.]
#    [0. 0. 0. ... 0. 0. 0.]
#    ...
#    [0. 0. 0. ... 0. 0. 0.]
#    [0. 0. 0. ... 0. 0. 0.]
#    [0. 0. 0. ... 0. 0. 0.]]]]
# (10000, 1, 28, 28)
# t_test: [7 2 1 ... 4 5 6]
# (10000, 1, 28, 28)
# *****************************************************************
# x_train: [[0. 0. 0. ... 0. 0. 0.]
#  [0. 0. 0. ... 0. 0. 0.]
#  [0. 0. 0. ... 0. 0. 0.]
#  ...
#  [0. 0. 0. ... 0. 0. 0.]
#  [0. 0. 0. ... 0. 0. 0.]
#  [0. 0. 0. ... 0. 0. 0.]]
# (60000, 784)
# t_train: [[0. 0. 0. ... 0. 0. 0.]
#  [1. 0. 0. ... 0. 0. 0.]
#  [0. 0. 0. ... 0. 0. 0.]
#  ...
#  [0. 0. 0. ... 0. 0. 0.]
#  [0. 0. 0. ... 0. 0. 0.]
#  [0. 0. 0. ... 0. 1. 0.]]
# (60000, 10)
# x_test: [[0. 0. 0. ... 0. 0. 0.]
#  [0. 0. 0. ... 0. 0. 0.]
#  [0. 0. 0. ... 0. 0. 0.]
#  ...
#  [0. 0. 0. ... 0. 0. 0.]
#  [0. 0. 0. ... 0. 0. 0.]
#  [0. 0. 0. ... 0. 0. 0.]]
# (10000, 784)
# t_test: [[0. 0. 0. ... 1. 0. 0.]
#  [0. 0. 1. ... 0. 0. 0.]
#  [0. 1. 0. ... 0. 0. 0.]
#  ...
#  [0. 0. 0. ... 0. 0. 0.]
#  [0. 0. 0. ... 0. 0. 0.]
#  [0. 0. 0. ... 0. 0. 0.]]
# (10000, 784)
# *****************************************************************
# $
