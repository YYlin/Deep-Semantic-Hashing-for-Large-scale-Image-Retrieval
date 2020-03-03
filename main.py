# -*- coding: utf-8 -*-
# @Time    : 2020/3/3 15:31
# @Author  : YYLin
# @Email   : 854280599@qq.com
# @File    : main.py
import os, h5py
import numpy as np
from keras.layers import Lambda, Input
from keras.applications.vgg19 import VGG19
from keras import regularizers
import keras, cv2
from sklearn.model_selection import train_test_split
from keras.models import Model
import warnings, sys
warnings.filterwarnings('ignore')
from keras import optimizers
from keras.datasets import mnist, cifar10

# 定义一些超参
batch_size = 128
img_size = 64
epoch = 10
learn_rate = 0.0001

# 加载数据集 mnist cifar10
dataset = 'mnist'
if dataset == 'mnist':
    print('***** loading mnist  *****')
    (train, train_y), (test, test_y) = mnist.load_data()
    train_y = keras.utils.to_categorical(train_y, 10)
    test_y = keras.utils.to_categorical(test_y, 10)

    # 修改测试集图像的大小
    train = [cv2.cvtColor(cv2.resize(i, (img_size, img_size)), cv2.COLOR_GRAY2RGB) for i in train]
    train = np.concatenate([arr[np.newaxis] for arr in train]).astype('float32')

    test = [cv2.cvtColor(cv2.resize(i, (img_size, img_size)), cv2.COLOR_GRAY2RGB) for i in test]
    test = np.concatenate([arr[np.newaxis] for arr in test]).astype('float32')

else:
    print('***** loading cifar10  *****')
    (train, train_y), (test, test_y) = cifar10.load_data()
    train_y = keras.utils.to_categorical(train_y, 10)
    test_y = keras.utils.to_categorical(test_y, 10)

    # 修改测试集图像的大小
    train = [cv2.resize(i, (img_size, img_size)) for i in train]
    train = np.concatenate([arr[np.newaxis] for arr in train]).astype('float32')

    test = [cv2.resize(i, (img_size, img_size)) for i in test]
    test = np.concatenate([arr[np.newaxis] for arr in test]).astype('float32')

print('train, train_y, test, test_y', train.shape, train_y.shape, test.shape, test_y.shape)


# 开始定义自己的训练模型
def gener_fea(model, preprocess=None, name=''):
    x = Input((img_size, img_size, 3))
    if preprocess:
        x = Lambda(preprocess)(x)
    base_model = model(input_tensor=x, weights='imagenet', include_top=False, pooling='avg')

    train_fea = base_model.predict(train, batch_size=batch_size)
    test_fea = base_model.predict(test, batch_size=batch_size)

    # 如果文件不存在的话 则将数据写入到模型之中
    if os.path.exists("%s_%s.h5" % (name, dataset)):
        print("%s_%s.h5" % (name, dataset), '已存在，不执行写操作')
    else:
        with h5py.File("%s_%s.h5" % (name, dataset), 'w') as f:
            print('正在保存数据..%s' % (name))
            f.create_dataset('train_fea', data=train_fea)
            f.create_dataset('train_y', data=train_y)
            f.create_dataset('train', data=train)

            f.create_dataset('test_fea', data=test_fea)
            f.create_dataset('test_y', data=test_y)
            f.create_dataset('test', data=test)

            f.close()
    return train_fea, test_fea


for_train, for_test = gener_fea(VGG19, name='VGG19')
X_train, X_val, y_train, y_val = train_test_split(for_train, train_y, shuffle=True, test_size=0.2, random_state=2019)

# 对于全连接层使用正则化以及dropout
inputs = Input((512,))
x = inputs
x = keras.layers.Dense(1024, kernel_regularizer=regularizers.l2(0.001), activation='relu')(x)

use_dropout = True  # 如果想使用dropout直接设置其为True 不想使用时设置为False
if use_dropout:
    x = keras.layers.Dropout(0.5)(x)

y = keras.layers.Dense((10), activation='softmax')(x)
model_1 = Model(inputs=inputs, outputs=y, name='Fusion')
optimi = optimizers.Adam(lr=learn_rate, beta_1=0.9, beta_2=0.95, epsilon=1e-08, decay=0.0)

# 编译模型并保存
model_1.compile(optimizer=optimi, loss='categorical_crossentropy', metrics=['accuracy'])
model_1.fit(x=X_train, y=y_train, batch_size=batch_size, epochs=epoch, validation_data=(X_val, y_val), verbose=1)
model_1.save('model_on_vgg_%s.h5'%dataset)
