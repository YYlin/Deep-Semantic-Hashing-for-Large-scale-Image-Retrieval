# -*- coding: utf-8 -*-
# @Time    : 2020/3/3 15:35
# @Author  : YYLin
# @Email   : 854280599@qq.com
# @File    : function_cell.py
import time
import numpy as np


# the hamming distance between two code
def hamming_distance(fisrt_num, second_num):

    hamming_distance = 0

    for index, img1_pix in enumerate(fisrt_num):
        img2_pix = second_num[index]
        if img1_pix != img2_pix:
            hamming_distance += 1
    return hamming_distance


# If the latter element is larger than the former in the list, add 1 or add 0
def get_diver(feats_list):
    print(len(feats_list), len(feats_list[0]))
    new_feats_list = []
    tmp = []
    for i, tmp_feat in enumerate(feats_list):
        for j in range(len(tmp_feat)-1):
            if tmp_feat[j] > tmp_feat[j + 1]:
                tmp.append(1)
            else:
                tmp.append(0)

        new_feats_list.append(tmp)
        tmp = []
    print('new_feats_list:', len(new_feats_list), len(new_feats_list[0]))
    return new_feats_list


# Convert binary code to hexadecimal code
def Binary_to_Hex(feats_lists, bit=16):
    decimal_value = 0
    hex_list = []
    hash_string = ""

    min_list_len = 200
    for i, feats in enumerate(feats_lists):
        for j, feat in enumerate(feats):
            if feat:  # value为0, 不用计算, 程序优化
                decimal_value += feat * (2 ** (j % bit))
            if j % bit == (bit-1):
                hash_string += str(hex(decimal_value)[2:].rjust(2, "0"))  # 不足2位以0填充。0xf=>0x0f
                decimal_value = 0

        if len(hash_string) < min_list_len:
            min_list_len = len(hash_string)

        hex_list.append(hash_string)

        hash_string = ''
    return hex_list, min_list_len


def getNowTime():
    return '['+time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))+']'


# change rgb into gray
def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])
