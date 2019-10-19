from os import listdir
import numpy as np
from scipy import ndimage
import Provider


subsetDirb= 'subset'
resampledSubsetDir = 'resample'
cropSubsetDir = 'crop'
candidates = 'CSVFILES/candidates_V2.csv'
RESIZE_SPACING = [1, 1, 1]
voxelWidthXY = 36
voxelWidthZ  = 24


if __name__ == '__main__':
    candidatesList = Provider.readCSV(candidates)
    for i in range(10):
        subsetDirn=subsetDirb+str(i)
        subsetDir=subsetDirb + '/' + subsetDirn
        list = listdir(subsetDir)
        subsetOuterList = []
        for file in list:
            if file.endswith(".mhd"):
                file = file[:-4]
                subsetOuterList.append(file)
        count0 = 0
        count1=0
        label = []
        for cand in candidatesList:
            if (cand[0] in subsetOuterList):
                fileName = cand[0] + '.mhd'
                originalFilePath = subsetDir + '/' + fileName
                newFilePath = resampledSubsetDir + '/' +subsetDirn+'/'+ cand[0] + '.npy'
                volumeImage, numpyOrigin, numpySpacing = Provider.load_itk_image(originalFilePath)
                newVolume = np.load(newFilePath)

                voxelWorldCoor = np.asarray([float(cand[3]), float(cand[2]), float(cand[1])])
                newGeneratedCoor = Provider.worldToVoxelCoord(voxelWorldCoor, numpyOrigin, RESIZE_SPACING)
                patch = newVolume[
                        int(newGeneratedCoor[0] - voxelWidthZ / 2):int(newGeneratedCoor[0] + voxelWidthZ / 2),
                        int(newGeneratedCoor[1] - voxelWidthXY / 2):int(newGeneratedCoor[1] + voxelWidthXY / 2),
                        int(newGeneratedCoor[2] - voxelWidthXY / 2):int(newGeneratedCoor[2] + voxelWidthXY / 2)]
                try :
                    if np.shape(patch) != (voxelWidthZ, voxelWidthXY, voxelWidthXY):
                        zoomFactor = [voxelWidthZ / float(np.shape(patch)[0]),
                                      voxelWidthXY / float(np.shape(patch)[1]),
                                      voxelWidthXY / float(np.shape(patch)[2])]
                        patch = ndimage.zoom(patch, zoom=zoomFactor)
                except Exception as e:
                    continue
                if cand[4]=='0':
                    np.save(cropSubsetDir + '/' +subsetDirn+'/'+'0'+'/'+ str(count), patch)
                    count0 += 1
                else:
                    np.save(cropSubsetDir + '/' + subsetDirn + '/' + '1' + '/' + str(count), patch)
                    count1 += 1


