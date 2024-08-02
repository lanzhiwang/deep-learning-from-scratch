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

url_base = 'https://ossci-datasets.s3.amazonaws.com/mnist/'  # mirror site
key_file = {
    'train_img': 'train-images-idx3-ubyte.gz',
    'train_label': 'train-labels-idx1-ubyte.gz',
    'test_img': 't10k-images-idx3-ubyte.gz',
    'test_label': 't10k-labels-idx1-ubyte.gz'
}

dataset_dir = os.path.dirname(os.path.abspath(__file__))
# print("dataset_dir:", dataset_dir)
# dataset_dir: /workspaces/deep-learning-from-scratch/dataset

save_file = dataset_dir + "/mnist.pkl"
# print("save_file:", save_file)
# save_file: /workspaces/deep-learning-from-scratch/dataset/mnist.pkl

train_num = 60000
test_num = 10000
img_dim = (1, 28, 28)
img_size = 784


def _download(file_name):
    # print("_download begin")
    # print("_download file_name:", file_name)
    # _download file_name: train-images-idx3-ubyte.gz
    # _download file_name: train-labels-idx1-ubyte.gz
    # _download file_name: t10k-images-idx3-ubyte.gz
    # _download file_name: t10k-labels-idx1-ubyte.gz
    file_path = dataset_dir + "/" + file_name
    # print("_download file_path:", file_path)
    # _download file_path: /workspaces/deep-learning-from-scratch/dataset/train-images-idx3-ubyte.gz
    # _download file_path: /workspaces/deep-learning-from-scratch/dataset/train-labels-idx1-ubyte.gz
    # _download file_path: /workspaces/deep-learning-from-scratch/dataset/t10k-images-idx3-ubyte.gz
    # _download file_path: /workspaces/deep-learning-from-scratch/dataset/t10k-labels-idx1-ubyte.gz

    if os.path.exists(file_path):
        return

    print("Downloading " + file_name + " ... ")
    # print("_download url_base + file_name:", url_base + file_name)
    # _download url_base + file_name: https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz
    # _download url_base + file_name: https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz
    # _download url_base + file_name: https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz
    # _download url_base + file_name: https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz
    urllib.request.urlretrieve(url_base + file_name, file_path)
    print("Done")
    # print("_download end")


def download_mnist():
    # print("download_mnist begin")
    for v in key_file.values():
        # print("download_mnist v:", v)
        # download_mnist v: train-images-idx3-ubyte.gz
        # download_mnist v: train-labels-idx1-ubyte.gz
        # download_mnist v: t10k-images-idx3-ubyte.gz
        # download_mnist v: t10k-labels-idx1-ubyte.gz
        _download(v)
    # print("download_mnist end")


def _load_label(file_name):
    file_path = dataset_dir + "/" + file_name

    print("Converting " + file_name + " to NumPy Array ...")
    with gzip.open(file_path, 'rb') as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8)
    print("Done")

    return labels


def _load_img(file_name):
    file_path = dataset_dir + "/" + file_name

    print("Converting " + file_name + " to NumPy Array ...")
    with gzip.open(file_path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    data = data.reshape(-1, img_size)
    print("Done")

    return data


def _convert_numpy():
    # print("_convert_numpy begin")
    dataset = {}
    dataset['train_img'] = _load_img(key_file['train_img'])
    dataset['train_label'] = _load_label(key_file['train_label'])
    dataset['test_img'] = _load_img(key_file['test_img'])
    dataset['test_label'] = _load_label(key_file['test_label'])
    # print("_convert_numpy dataset:", dataset)
    # _convert_numpy dataset: {'train_img': array([[0, 0, 0, ..., 0, 0, 0],
    #    [0, 0, 0, ..., 0, 0, 0],
    #    [0, 0, 0, ..., 0, 0, 0],
    #    ...,
    #    [0, 0, 0, ..., 0, 0, 0],
    #    [0, 0, 0, ..., 0, 0, 0],
    #    [0, 0, 0, ..., 0, 0, 0]], dtype=uint8), 'train_label': array([5, 0, 4, ..., 5, 6, 8], dtype=uint8), 'test_img': array([[0, 0, 0, ..., 0, 0, 0],
    #    [0, 0, 0, ..., 0, 0, 0],
    #    [0, 0, 0, ..., 0, 0, 0],
    #    ...,
    #    [0, 0, 0, ..., 0, 0, 0],
    #    [0, 0, 0, ..., 0, 0, 0],
    #    [0, 0, 0, ..., 0, 0, 0]], dtype=uint8), 'test_label': array([7, 2, 1, ..., 4, 5, 6], dtype=uint8)}
    # print("_convert_numpy end")

    return dataset


def init_mnist():
    # print("init_mnist begin")
    download_mnist()
    dataset = _convert_numpy()
    print("Creating pickle file ...")
    with open(save_file, 'wb') as f:
        pickle.dump(dataset, f, -1)
    print("Done!")
    # print("init_mnist end")


def _change_one_hot_label(X):
    T = np.zeros((X.size, 10))
    for idx, row in enumerate(T):
        row[X[idx]] = 1

    return T


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
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].astype(np.float32)
            dataset[key] /= 255.0

    if one_hot_label:
        dataset['train_label'] = _change_one_hot_label(dataset['train_label'])
        dataset['test_label'] = _change_one_hot_label(dataset['test_label'])

    if not flatten:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].reshape(-1, 1, 28, 28)

    return (dataset['train_img'],
            dataset['train_label']), (dataset['test_img'],
                                      dataset['test_label'])


if __name__ == '__main__':
    init_mnist()
    load_mnist(normalize=True, flatten=False, one_hot_label=True)
