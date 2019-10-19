# -*- coding:utf-8 -*-
'''
this script is used for basic process of lung 2017 in Data Science Bowl
'''
import glob
import os
import pandas as pd
import SimpleITK as sitk
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import skimage, os
from skimage.morphology import ball, disk, dilation, binary_erosion, remove_small_objects, erosion, closing, \
    reconstruction, binary_closing
from skimage.measure import label, regionprops, perimeter
from skimage.morphology import binary_dilation, binary_opening
from skimage.filters import roberts, sobel
from skimage import measure, feature
from skimage.segmentation import clear_border
from skimage import data
from scipy import ndimage as ndi
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pydicom
import scipy.misc
import numpy as np
from os import listdir
import numpy as np
from scipy import ndimage

# Directories
baseSubsetDirectory = r'E:/JLS/dcm_data/luna/subsets/subset'

RESIZE_SPACING = [1, 1, 1]
NUM_SUBSET = 10


def load_itk_image(filename):
    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage)
    numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
    numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))

    return numpyImage, numpyOrigin, numpySpacing


def get_segmented_lungs(im, plot=False):
    '''
    This funtion segments the lungs from the given 2D slice.
    '''
    # if plot == True:
    #     f, plots = plt.subplots(4, 2, figsize=(5, 40))
    '''
    Step 1: Convert into a binary image.
    '''
    binary = im < -600
    # if plot == True:
    #     plots[0][0].axis('off')
    #     plots[0][0].set_title('binary image')
    #     plots[0][0].imshow(binary, cmap=plt.cm.bone)
    '''
    Step 2: Remove the blobs connected to the border of the image.
    '''
    cleared = clear_border(binary)
    # if plot == True:
    #     plots[0][1].axis('off')
    #     plots[0][1].set_title('after clear border')
    #     plots[0][1].imshow(cleared, cmap=plt.cm.bone)
    '''
    Step 3: Label the image.
    '''
    label_image = label(cleared)
    # if plot == True:
    #     plots[1][0].axis('off')
    #     plots[1][0].set_title('found all connective graph')
    #     plots[1][0].imshow(label_image, cmap=plt.cm.bone)
    '''
    Step 4: Keep the labels with 2 largest areas.
    '''
    areas = [r.area for r in regionprops(label_image)]
    areas.sort()
    if len(areas) > 2:
        for region in regionprops(label_image):
            if region.area < areas[-2]:
                for coordinates in region.coords:
                    label_image[coordinates[0], coordinates[1]] = 0
    binary = label_image > 0
    # if plot == True:
    #     plots[1][1].axis('off')
    #     plots[1][1].set_title(' Keep the labels with 2 largest areas')
    #     plots[1][1].imshow(binary, cmap=plt.cm.bone)
    '''
    Step 5: Erosion operation with a disk of radius 2. This operation is
    seperate the lung nodules attached to the blood vessels.
    '''
    selem = disk(2)
    binary = binary_erosion(binary, selem)
    # if plot == True:
    #     plots[2][0].axis('off')
    #     plots[2][0].set_title('seperate the lung nodules attached to the blood vessels')
    #     plots[2][0].imshow(binary, cmap=plt.cm.bone)
    '''
    Step 6: Closure operation with a disk of radius 10. This operation is
    to keep nodules attached to the lung wall.
    '''
    selem = disk(10)
    binary = binary_closing(binary, selem)
    # if plot == True:
    #     plots[2][1].axis('off')
    #     plots[2][1].set_title('keep nodules attached to the lung wall')
    #     plots[2][1].imshow(binary, cmap=plt.cm.bone)
    '''
    Step 7: Fill in the small holes inside the binary mask of lungs.
    '''
    edges = roberts(binary)
    binary = ndi.binary_fill_holes(edges)
    # if plot == True:
    #     plots[3][0].axis('off')
    #     plots[3][0].set_title('Fill in the small holes inside the binary mask of lungs')
    #     plots[3][0].imshow(binary, cmap=plt.cm.bone)
    '''
    Step 8: Superimpose the binary mask on the input image.
    '''
    sum = 0

    for r in regionprops(label_image):
        sum = sum + r.area
    proportion = sum / (512 * 512)

    im = (255 / 1800 * im + 1200 * 255 / 1800)
    get_high_vals = binary == 0
    im[get_high_vals] = 170
    im = np.rint(im)
    # if plot == True:
    #     plots[3][1].axis('off')
    #     plots[3][1].set_title('Superimpose the binary mask on the input image')
    #     plots[3][1].imshow(im, cmap=plt.cm.bone)
    return im, proportion


def plot_ct_scan(scan):
    '''
            plot a few more images of the slices
    :param scan:
    :return:
    '''
    f, plots = plt.subplots(int(scan.shape[0] / 20) + 1, 4, figsize=(50, 50))
    for i in range(0, scan.shape[0], 5):
        plots[int(i / 20), int((i % 20) / 5)].axis('off')
        plots[int(i / 20), int((i % 20) / 5)].imshow(scan[i])


def mysum(arr, sum_number):
    if len(arr) < sum_number:
        return -1
    else:
        pos = 0
        beforeTotal = 0
        for i in range(len(arr) - sum_number):
            Total = np.sum(arr[i:i + sum_number])
            if Total > beforeTotal:
                beforeTotal = Total
                pos = i
    return pos


filePath = 'E:\\JLS\\dcm_data\\luna\\subsets\\subset1\\1.3.6.1.4.1.14519.5.2.1.6279.6001.173106154739244262091404659845.mhd'
volumeImage, numpyOrigin, numpySpacing = load_itk_image(filePath)

resize_factor = numpySpacing / RESIZE_SPACING
new_real_shape = volumeImage.shape * resize_factor
# volumeImage[volumeImage <= -600]= 0
# volumeImage[volumeImage > -600] = 1
new_shape = np.round(new_real_shape)

real_resize = new_shape / volumeImage.shape

new_volume = ndimage.zoom(volumeImage, zoom=real_resize)
LP = []  # lung proportion
resample_im = []

for idx in range(len(new_volume)):
    # print(len(new_volume))
    # plot_ct_scan(new_volume)
    data = new_volume[idx]
    # print(data[314][303])
    # plt.figure(100)
    # plt.imshow(data, cmap='gray')
    im, proportion = get_segmented_lungs(data, plot=True)
    LP.append(proportion)
    resample_im.append(im)
    # print(im[7][7])
    # print(im[314][303],np.max(np.max(im)),np.min(np.min(im)))
    # plt.figure(200)
    # plt.imshow(im, cmap='gray')
# plt.show()
pos = mysum(LP, 128)
FIM = resample_im[pos:pos + 128]
FIM = np.array(FIM)
int(FIM.shape[1] / 2) - 64
int(FIM.shape[1] / 2) - 128 - 15
FIM = np.array(FIM[0:128, int(FIM.shape[1] / 2) - 64:int(FIM.shape[1] / 2) + 64,
               int(FIM.shape[1] / 2) - 128 - 15:int(FIM.shape[1] / 2) - 15])
data = FIM[0]
np.save('test.npy', FIM)
plt.figure(200)
plt.imshow(data, cmap='gray')
# plot_ct_scan(FIM)
plt.show()
最后保存的test.npy
就是这128 * 128 * 128（zxy）的立方体数据。

显示最终结果代码如下：

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

newFilePath1 = 'test.npy'
aa = np.array([1, 2, 3, 5])
print(aa[0:2])
newVolume1 = np.load(newFilePath1)
print(len(newVolume1))
for i in range(len(newVolume1)):
    if i % 5 == 0:
        print(i)
        data = newVolume1[i]
        plt.figure(i)
        plt.imshow(data, cmap='gray')
plt.show()
