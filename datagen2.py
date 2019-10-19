from Provider import *
import os
import numpy as np
import keras
import math




subsetDirb= 'subset'
resampledSubsetDir = 'resample'
cropSubsetDir = 'crop'
candidates = 'CSVFILES/candidates_V2.csv'
RESIZE_SPACING = [1, 1, 1]
voxelWidthXY = 36
voxelWidthZ  = 24


class DataGenerator(keras.utils.Sequence):
    def __init__(self, paths, batch_size=128):
        self.batch_size = batch_size
        self.paths = paths
        self.indexes = np.arange(len(self.paths))

    def __len__(self):
        # 计算每一个epoch的迭代次数
        return math.ceil(len(self.paths) / float(self.batch_size))

    def __getitem__(self, index):
        # 生成每个batch数据，这里就根据自己对数据的读取方式进行发挥了
        # 生成batch_size个索引
        batch_indexs = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        # 根据索引获取datas集合中的数据
        batch_paths = [self.paths[k] for k in batch_indexs]
        # 生成数据
        X, y = self.data_generation(batch_paths)

        return X, y

    def data_generation(self, batch_paths):
        datas = []
        labels = []

        # 生成数据
        for path in batch_paths:
            # x_train数据
            data = np.load(path)
            data = dataNormalize(data)
            data = np.reshape(data, (24, 36, 36, 1))
            datas.append(data)
            # y_train数据
            right = path.rfind("/", 0)
            left = path.rfind("/", 0, right) + 1
            class_name = path[left:right]
            if class_name == "0":
                labels.append(0)
            else:
                labels.append(1)
        # 如果为多输出模型，Y的格式要变一下，外层list格式包裹numpy格式是list[numpy_out1,numpy_out2,numpy_out3]
        return np.array(datas), np.array(labels)


def test_generate(k):
    test_paths = []
    datapath = "crop/subset"+str(k)
    fpath = os.path.join(datapath, '0')
    tpath = os.path.join(datapath, '1')
    for file in os.listdir(fpath):
        test_paths.append(os.path.join(fpath, file))
    for file in os.listdir(tpath):
        test_paths.append(os.path.join(tpath, file))

    regen = DataGenerator(test_paths)

    return regen



def label_generator(k):
    labels=[]
    test_paths = []
    datapath = "crop/subset"+str(k)
    fpath = os.path.join(datapath, '0')
    tpath = os.path.join(datapath, '1')
    for file in os.listdir(fpath):
        test_paths.append(os.path.join(fpath, file))
    for file in os.listdir(tpath):
        test_paths.append(os.path.join(tpath, file))
    for path in test_paths:
        right = path.rfind("/", 0)
        left = path.rfind("/", 0, right) + 1
        class_name = path[left:right]
        if class_name == "0":
            labels.append(0.0)
        else:
            labels.append(1.0)

    return labels
