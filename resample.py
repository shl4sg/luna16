from os import listdir
import numpy as np
import Provider
from scipy import ndimage

# Directories
subsetDirb = 'subset'
targetSubsetDir = 'resample'
RESIZE_SPACING = [1,1,1]
if __name__ == '__main__':
    for i in range(10):
        subsetDirn = subsetDirb + str(i)
        subsetDir = subsetDirb + '/' + subsetDirn
        list = listdir(subsetDir)
        subsetList = []
        # Create Subset List
        for file in list:
            if file.endswith(".mhd"):
                subsetList.append(file)
        for file in subsetList:
            fileName = file[:-4]
            filePath = subsetDir + '/' + file
            volumeImage, numpyOrigin, numpySpacing = Provider.load_itk_image(filePath)
            resize_factor = numpySpacing / RESIZE_SPACING
            new_real_shape = volumeImage.shape * resize_factor
            new_shape = np.round(new_real_shape)
            real_resize = new_shape / volumeImage.shape
            new_volume = ndimage.zoom(volumeImage, zoom=real_resize)
            np.save(targetSubsetDir + '/' + subsetDirn + '/' + fileName, new_volume)




