from remodel.sedensenet2 import*
from remodel.alexnet import *
from keras import backend as K
from datagen import *
from keras.callbacks import ModelCheckpoint
from FROC import *
from keras.models import load_model



for i in range(10):
    print('start '+str(i))
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)
    # 训练集生成
    trainGene = train_generate(i)
    # 模型生成
    model = se_densenet()
    model_checkpoint = ModelCheckpoint(str(i)+'sd.hdf5', monitor='loss', verbose=1, save_best_only=True)
    # 模型训练
    model.fit_generator(trainGene, steps_per_epoch=len(trainGene), epochs=20, callbacks=[model_checkpoint])
    # 测试集生成
    # testGene = test_generate(i)
    # 评估
    # loss, accuracy = model.evaluate_generator(testGene, 30, verbose=1)
    # print(loss, accuracy)

