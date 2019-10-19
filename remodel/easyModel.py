
from keras.models import *
from keras.layers import *
from keras.optimizers import *

voxelWidthXY = 36
voxelWidthZ  = 24


def easynet(pretrained_weights=None, input_size=(24,36,36,1)):
    inputs = Input(input_size)
    conv1 = Conv3D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    pool1 = MaxPooling3D(pool_size=(3))(conv1)
    drop1 = Dropout(0.2)(pool1)

    conv2 = Conv3D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(drop1)
    pool2 = MaxPooling3D(pool_size=(3))(conv2)
    drop2 = Dropout(0.2)(pool2)

    flatten=Flatten()(drop2)
    dense3 = Dense(256, activation='relu')(flatten)
    drop3 = Dropout(0.2)(dense3)
    dense4=Dense(1,activation='sigmoid')(drop3)

    model = Model(input=inputs, output=dense4)
    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    # remodel.summary()
    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model