# -*- coding: utf-8 -*-
# @Time    : 2020/3/3 15:32
# @Author  : YYLin
# @Email   : 854280599@qq.com
# @File    : Calculation_MAP.py
from function_cell import *
import h5py
import cv2
import numpy as np
import os

# Dataload VGG19_mnist VGG19_cifar10
dataset_name = 'VGG19_cifar10'
dataset_path = '%s.h5'%dataset_name
h5 = h5py.File(dataset_path, 'r')
train_fea = h5['train_fea'][:]
train_y = h5['train_y'][:]
train_img = h5['train'][:]

test_fea = h5['test_fea'][:]
test_y = h5['test_y'][:]
test_img = h5['test'][:]
h5.close()

seed = 2019
np.random.seed(seed)
np.random.shuffle(train_fea)

np.random.seed(seed)
np.random.shuffle(train_y)

np.random.seed(seed)
np.random.shuffle(train_img)

seed_2 = 2020
np.random.seed(seed_2)
np.random.shuffle(test_fea)

np.random.seed(seed_2)
np.random.shuffle(test_y)

np.random.seed(seed_2)
np.random.shuffle(test_img)

bit_list = [8, 16, 24]
for bit in bit_list:
    print('Begin change the feature of Train dataset into hash code:', getNowTime())
    diver_of_train = get_diver(train_fea.tolist())
    hex_of_train, train_min_hex_len = Binary_to_Hex(diver_of_train, bit=bit)
    print('Finish change the feature of Train dataset into hash code:', getNowTime())

    print('Begin change the feature of Test dataset into hash code:', getNowTime())
    diver_of_test = get_diver(test_fea.tolist())
    hex_of_test, test_min_hex_len = Binary_to_Hex(diver_of_test, bit=bit)
    print('Finish change the feature of Test dataset into hash code:', getNowTime())

    # Some super parameters for calculating MAP
    return_of_top_nn = 5000
    len_of_test = 100
    MAP = 0

    if train_min_hex_len > test_min_hex_len:
        train_min_hex_len = test_min_hex_len
    print('the samllest length in train and test dataset is:', train_min_hex_len)

    begin = time.time()

    for i, hex_tests in enumerate(hex_of_test[0:len_of_test]):
        AP = 0
        AP_Len = 0
        hanming_list = []
        for j, train_hex in enumerate(hex_of_train):
            tmp_hanming = hamming_distance(hex_tests[0:train_min_hex_len], train_hex[0:train_min_hex_len])
            hanming_list.append(tmp_hanming)

        #  If the retrieve images and retrieved images have the same categories, the retrieval is successful
        top_hanming_list = np.argsort(hanming_list)[0:return_of_top_nn]
        for j, tmp in enumerate(top_hanming_list):
            # Save the first 10 similar images
            if i == 0:
                if not os.path.exists(dataset_name):
                    os.makedirs(dataset_name)
                cv2.imwrite('%s/old_img_%d.png' % (dataset_name, i), test_img[i])
                cv2.imwrite('%s/retrieval_img_%d_%d_%d.png' % (dataset_name, i, j, tmp), train_img[tmp])

            if (train_y[tmp] == test_y[i]).all():
                # print(train_y[tmp], test_y[i])
                AP_Len += 1
                AP += j / return_of_top_nn
        MAP += (AP/AP_Len)

    end = time.time()
    print('Finish image retrieval:', getNowTime())
    print('The MAP on bit %d is:' % bit, MAP / len_of_test, 'the time is:', end - begin)





