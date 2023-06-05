# coding: utf-8
import sys, os

sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import numpy as np
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True,
                                                  one_hot_label=True)
print(x_train.shape)  # (60000, 784)
print(t_train.shape)  # (60000, 10)

train_size = x_train.shape[0]
print(train_size)  # 60000

batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)
print(batch_mask)
# [11972  2315  8434 51555  6663 57849 40158 55868 28095 56004]

x_batch = x_train[batch_mask]
print(x_batch.shape)  # (10, 784)

t_batch = t_train[batch_mask]
print(t_batch.shape)  # (10, 10)
