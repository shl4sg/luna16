from datagen2 import *
from remodel.alexnet import *
from keras import backend as K
from focalloss import binary_focal_loss
from keras.utils.generic_utils import get_custom_objects

loss=binary_focal_loss(alpha=.25, gamma=2)
get_custom_objects().update({'binary_focal_loss_fixed': loss})
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.set_session(sess)



for i in range(10):
    model = load_model(str(i) + 'sd.hdf5')
    # 预测和FROC
    testGene = test_generate(i)
    labels = label_generator(i)
    steps=len(testGene)

    predict= model.predict_generator(testGene, steps=steps, verbose=1)
    predict=np.array(predict)
    labels=np.array(labels)
    dir = 'data/pre' + str(i)
    np.save(dir,predict)
    dir = 'data/label' + str(i)
    np.save(dir,labels)


