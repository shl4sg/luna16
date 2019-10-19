
from keras.models import *
from keras.layers import *
from keras.optimizers import *

voxelWidthXY = 36
voxelWidthZ  = 24

def alexnet(input_size=(24,36,36,1)):
    inputs = Input(input_size)
    conv1 = Conv3D(64, kernel_size=(3,5,5), activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    pool1 = MaxPooling3D(pool_size=(2, 2,2))(conv1)
    conv2 = Conv3D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    pool2 = MaxPooling3D(pool_size=(2, 2,2))(conv2)

    conv3 = Conv3D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv3D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3=MaxPooling3D(pool_size=(2, 2,2))(conv3)

    conv4=Conv3D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv3D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    pool4=MaxPooling3D(pool_size=(2, 2,2))(conv4)

    flatten=Flatten()(pool4)
    dense4= Dense(512,activation='relu')(flatten)
    dense4=Dense(128,activation='relu')(dense4)
    dense5=Dense(1,activation='sigmoid')(dense4)
    model = Model(input=inputs, output=dense5)
    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    return model
