from keras.models import *
from keras.layers import *
from remodel.senet import *
from keras.optimizers import *
from focalloss import *

voxelWidthXY = 36
voxelWidthZ  = 24
growth_rate=[12,12,12,24]


def se_conv_block(x, growth_rate,k, name):
    """
    A building block for a dense block.
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
    bn_axis =4
    x1 = Conv3D(growth_rate, 3, padding='same', use_bias=False,name=name + '_1_conv', kernel_initializer='he_normal')(x)
    x1 = Activation('relu', name=name + '_0_relu', kernel_initializer='he_normal')(x1)
    x1 = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(x1)
    x1 = squeeze_excitation_layer(x1,growth_rate,4)
    x = Concatenate(axis=bn_axis, name=name + '_concat')([x, x1])
    if k!=3:
        x = Conv3D(growth_rate, 1, use_bias=False, name=name + '_2_conv', kernel_initializer='he_normal')(x)
    return x

def se_dense_block(x, blocks,k, name):
    """A se-dense block.
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
        x = se_conv_block(x, growth_rate[k], k,name=name + '_block' + str(i + 1))
    return x



def se_densenet():
    blocks = [4, 6, 8, 10]

    input = Input((24, 36, 36,1), name='data')
    x = se_dense_block(input, blocks[0],0 , name='conv1')
    x = MaxPooling3D(2, strides=2, name='pool1')(x)
    x = se_dense_block(x, blocks[1],1, name='conv3')
    x = MaxPooling3D(2, strides=2, name= 'pool2')(x)
    x = se_dense_block(x, blocks[2],2, name='conv4')
    x = MaxPooling3D(2, strides=2, name='pool3')(x)
    x = se_dense_block(x, blocks[3],3, name='conv5')
    x = MaxPooling3D(2, strides=2, name='pool4')(x)
    flatten=Flatten()(x)
    output = Dense(1, activation='sigmoid', name='dense', kernel_initializer='he_normal')(flatten)
    model = Model(input, output, name='se_densenet')
    model.compile(optimizer=Adam(lr=1e-4), loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=['accuracy'])
    return model
