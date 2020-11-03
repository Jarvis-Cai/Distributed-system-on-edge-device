# -*- coding:utf-8 -*-
# Data reading and preprocessing module

import pickle
import random
import numpy as np

class_num = 10
image_size = 32
img_channels = 3


# ========================================================== #
# ├─ prepare_data()
#  ├─ download training data if not exist by download_data()
#  ├─ load data by load_data()
#  └─ shuffe and return data
# ========================================================== #

# Get the information in each batch file
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# read file data
def load_data_one(file):
    batch = unpickle(file)
    data = batch[b'data']                           # read picture information
    labels = batch[b'labels']                       # read labels information
    print("Loading %s : %d." % (file, len(data)))
    return data, labels

# no using this function
def load_data(files, data_dir, label_count):
    global image_size, img_channels
    data, labels = load_data_one(data_dir + '/' + files[0])
    for f in files[1:]:
        data_n, labels_n = load_data_one(data_dir + '/' + f)
        data = np.append(data, data_n, axis=0)
        labels = np.append(labels, labels_n, axis=0)
    labels = np.array([[float(i == label) for i in range(label_count)] for label in labels])
    data = data.reshape([-1, img_channels, image_size, image_size])
    data = data.transpose([0, 2, 3, 1])
    return data, labels

# read file list
def load_data_mine(files, data_dir, label_count):
    global image_size, img_channels
    data, labels = load_data_one(data_dir + '/' + files[0])         # data_batch_1 
    for f in files[1:]:
        data_n, labels_n = load_data_one(data_dir + '/' + f)        # data_batch_n 
        data = np.concatenate((data, data_n))
        labels = np.concatenate((labels, labels_n))

    num_data = len(labels)                                          # total data number
    labels = np.array([[float(i == label) for i in range(label_count)] for label in labels])
    data = data.reshape((num_data, image_size * image_size, img_channels), order='F')
    data = data.reshape((num_data, image_size, image_size, img_channels))
    return data, labels


def prepare_data():
    print("======Loading data======")
    data_dir = './cifar-10-batches-py'                      # Root Path
    image_dim = image_size * image_size * img_channels      # picture size 32*32*3 = 3072
    meta = unpickle(data_dir + '/batches.meta')

    label_names = meta[b'label_names']                      # label_names[0]=="airplane",label_names[1]=="automobile",etc...
    label_count = len(label_names)                          # 10 types
    train_files = ['data_batch_%d' % d for d in range(1, 6)]
    train_data, train_labels = load_data_mine(train_files, data_dir, label_count)   # read train data
    test_data, test_labels = load_data_mine(['test_batch'], data_dir, label_count)  # read test data

    print("Train data:", np.shape(train_data), np.shape(train_labels))
    print("Test data :", np.shape(test_data), np.shape(test_labels))
    print("======Load finished======")

    # Randomly get data
    print("======Shuffling data======")
    indices = np.random.permutation(len(train_data))        
    train_data = train_data[indices]
    train_labels = train_labels[indices]
    indices = np.random.permutation(len(test_data))
    test_data = test_data[indices]
    test_labels = test_labels[indices]


    print("======Prepare Finished======")

    return train_data, train_labels, test_data, test_labels


# ========================================================== #
# ├─ _random_crop()
# ├─ _random_flip_leftright()
# ├─ data_augmentation()
# └─ color_preprocessing()
# ========================================================== #
# is no using?
def _random_crop(batch, crop_shape, padding=None):
    oshape = np.shape(batch[0])

    if padding:
        oshape = (oshape[0] + 2 * padding, oshape[1] + 2 * padding)
    new_batch = []
    npad = ((padding, padding), (padding, padding), (0, 0))
    for i in range(len(batch)):
        new_batch.append(batch[i])
        if padding:
            new_batch[i] = np.lib.pad(batch[i], pad_width=npad,
                                      mode='constant', constant_values=0)
        nh = random.randint(0, oshape[0] - crop_shape[0])
        nw = random.randint(0, oshape[1] - crop_shape[1])
        new_batch[i] = new_batch[i][nh:nh + crop_shape[0],
                       nw:nw + crop_shape[1]]
    return new_batch


def _random_flip_leftright(batch):
    for i in range(len(batch)):
        if bool(random.getrandbits(1)):
            batch[i] = np.fliplr(batch[i])
    return batch


def color_preprocessing(x_train, x_val):
    x_train = x_train.astype('float32')
    x_val = x_val.astype('float32')
    x_train[:, :, :, 0] = (x_train[:, :, :, 0] - np.mean(x_train[:, :, :, 0])) / np.std(x_train[:, :, :, 0])
    x_train[:, :, :, 1] = (x_train[:, :, :, 1] - np.mean(x_train[:, :, :, 1])) / np.std(x_train[:, :, :, 1])
    x_train[:, :, :, 2] = (x_train[:, :, :, 2] - np.mean(x_train[:, :, :, 2])) / np.std(x_train[:, :, :, 2])

    x_val[:, :, :, 0] = (x_val[:, :, :, 0] - np.mean(x_val[:, :, :, 0])) / np.std(x_val[:, :, :, 0])
    x_val[:, :, :, 1] = (x_val[:, :, :, 1] - np.mean(x_val[:, :, :, 1])) / np.std(x_val[:, :, :, 1])
    x_val[:, :, :, 2] = (x_val[:, :, :, 2] - np.mean(x_val[:, :, :, 2])) / np.std(x_val[:, :, :, 2])

    return x_train, x_val


def data_augmentation(batch):
    batch = _random_flip_leftright(batch)
    batch = _random_crop(batch, [32, 32], 4)
    return batch


def guiyihua(x_train, x_test):
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train = x_train/255
    x_test = x_test/255
    return x_train, x_test
