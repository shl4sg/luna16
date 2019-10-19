

from keras.layers import *



def squeeze_excitation_layer(x, out_dim,ratio):
    '''
    SE module performs inter-channel weighting.
    '''
    squeeze = GlobalAveragePooling3D()(x)
    excitation = Dense(units=out_dim // ratio,activation='relu')(squeeze)
    excitation = Dense(units=out_dim,activation='sigmoid')(excitation)
    excitation = Reshape((1, 1,1, out_dim))(excitation)
    scale = multiply([x, excitation])
    return scale