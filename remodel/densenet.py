import keras
import numpy as np
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K

from keras.layers.normalization import BatchNormalization
import keras.backend as K


def dense_block(x, blocks, name):
    """A dense block.
    密集的模块
    # Arguments
    参数
        x: input tensor.
        x: 输入参数
        blocks: integer, the number of building blocks.
        blocks: 整型，生成块的个数。
        name: string, block label.
        name: 字符串，块的标签
    # Returns
    返回
        output tensor for the block.
        为块输出张量
    """
    for i in range(blocks):
        x = conv_block(x, 32, name=name + '_block' + str(i + 1))
    return x


def transition_block(x, reduction, name):
    """A transition block.
    转换块
    # Arguments
    参数
        x: input tensor.
        x: 输入参数
        reduction: float, compression rate at transition layers.
        reduction: 浮点数，转换层的压缩率
        name: string, block label.
        name: 字符串，块标签
    # Returns
    返回
        output tensor for the block.
        块输出张量
    """
    bn_axis = 3
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,name=name + '_bn')(x)
    x = Activation('relu', name=name + '_relu')(x)
    x = Conv2D(int(K.int_shape(x)[bn_axis] * reduction), 1, use_bias=False,name=name + '_conv')(x)
    x = AveragePooling2D(2, strides=2, name=name + '_pool')(x)
    return x


def conv_block(x, growth_rate, name,):
    """A building block for a dense block.
    密集块正在建立的块
    # Arguments
    参数
        x: input tensor.
        x: 输入张量
        growth_rate: float, growth rate at dense layers.
        growth_rate:浮点数，密集层的增长率。
        name: string, block label.
        name: 字符串，块标签
    # Returns
    返回
        output tensor for the block.
        块输出张量
    """
    bn_axis = 3
    x1 = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,name=name + '_0_bn')(x)
    x1 = Activation('relu', name=name + '_0_relu')(x1)
    x1 = Conv2D(4 * growth_rate, 1, use_bias=False,name=name + '_1_conv')(x1)
    x1 = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,name=name + '_1_bn')(x1)
    x1 = Activation('relu', name=name + '_1_relu')(x1)
    x1 = Conv2D(growth_rate, 3, padding='same', use_bias=False,name=name + '_2_conv')(x1)
    x = Concatenate(axis=bn_axis, name=name + '_concat')([x, x1])
    return x


def densenet(growth_rate=32,classes=2):
    blocks = [6, 12, 24, 16]
    bn_axis=3
    img_input = Input(shape=(224, 224, 3), name='data')
    x = ZeroPadding2D(padding=((3, 3), (3, 3)))(img_input)
    x = Conv2D(growth_rate * 2, 7, strides=2, use_bias=False, name='conv1/conv')(x)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name='conv1/bn')(x)
    x = Activation('relu', name='conv1/relu')(x)
    x = ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
    x = MaxPooling2D(3, strides=2, name='pool1')(x)

    x = dense_block(x, blocks[0], name='conv2')
    x = transition_block(x, 0.5, name='pool2')
    x = dense_block(x, blocks[1], name='conv3')
    x = transition_block(x, 0.5, name='pool3')
    x = dense_block(x, blocks[2], name='conv4')
    x = transition_block(x, 0.5, name='pool4')
    x = dense_block(x, blocks[3], name='conv5')

    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,name='bn')(x)
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dense(classes, activation='softmax', name='fc1000')(x)
    model = Model(img_input, x, name='densenet')
