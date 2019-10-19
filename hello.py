import Provider
from scipy import ndimage
import numpy as np
from os import listdir


subsetDirb= 'subset'
cropSubsetDir = 'crop'
candidates = 'CSVFILES/candidates_V2.csv'
RESIZE_SPACING = [1, 1, 1]
voxelWidthXY = 36
voxelWidthZ  = 24

if __name__ == '__main__':
    path='pre0.npy'
    data=np.load(path)
    print(data)



