from os import listdir
import numpy as np
from scipy import ndimage
from Provider import *

subsetDirb = 'subset'
resampledSubsetDir = 'resample'
cropSubsetDir = 'crop'
candidates = 'CSVFILES/candidates_V2.csv'
RESIZE_SPACING = [1, 1, 1]
voxelWidthXY = 36
voxelWidthZ = 24

if __name__ == '__main__':
    candidatesList = readCSV(candidates)
    for i in range(10):
        subsetDirn = subsetDirb + str(i)
        subsetDir = subsetDirb + '/' + subsetDirn
        list = listdir(subsetDir)
        subsetOuterList = []
        for file in list:
            if file.endswith(".mhd"):
                file = file[:-4]
                subsetOuterList.append(file)
        count = 0

        label = []
        for cand in candidatesList:
            if cand[4] == '1':
                if (cand[0] in subsetOuterList):
                    fileName = cand[0] + '.mhd'
                    originalFilePath = subsetDir + '/' + fileName
                    newFilePath = resampledSubsetDir + '/' + subsetDirn + '/' + cand[0] + '.npy'
                    volumeImage, numpyOrigin, numpySpacing = load_itk_image(originalFilePath)
                    newVolume = np.load(newFilePath)
                    voxelWorldCoor = np.asarray([float(cand[3]), float(cand[2]), float(cand[1])])
                    newGeneratedCoor = worldToVoxelCoord(voxelWorldCoor, numpyOrigin, RESIZE_SPACING)
                    patch = newVolume[
                            int(newGeneratedCoor[0] - voxelWidthZ / 2):int(newGeneratedCoor[0] + voxelWidthZ / 2),
                            int(newGeneratedCoor[1] - voxelWidthXY / 2):int(newGeneratedCoor[1] + voxelWidthXY / 2),
                            int(newGeneratedCoor[2] - voxelWidthXY / 2):int(newGeneratedCoor[2] + voxelWidthXY / 2)]
                    try:
                        if np.shape(patch) != (voxelWidthZ, voxelWidthXY, voxelWidthXY):
                            zoomFactor = [voxelWidthZ / float(np.shape(patch)[0]),
                                          voxelWidthXY / float(np.shape(patch)[1]),
                                          voxelWidthXY / float(np.shape(patch)[2])]
                            patch = ndimage.zoom(patch, zoom=zoomFactor)
                    except Exception:
                        continue

                    patchS, countS = dataShift(newGeneratedCoor, voxelWidthZ, voxelWidthXY, newVolume)
                    for patchn in range(countS):
                        np.save(cropSubsetDir + '/' + subsetDirn + '/' + '1a' + '/' + str(count), patchS[patchn])
                        count += 1
                    patchF,countF = dataFlip(patch)
                    for patchn in range(countF):
                        np.save(cropSubsetDir + '/' + subsetDirn + '/' + '1a' + '/' + str(count), patchF[patchn])
                        count += 1
                    patchR,countR = dataRot(patch,newGeneratedCoor, voxelWidthZ, voxelWidthXY, newVolume)
                    for patchn in range(countR):
                        np.save(cropSubsetDir + '/' + subsetDirn + '/' + '1a' + '/' + str(count), patchR[patchn])
                        count += 1
                    patchSF,countSF=dataSF(patchS,countS)
                    for patchn in range(countSF):
                        np.save(cropSubsetDir + '/' + subsetDirn + '/' + '1a' + '/' + str(count), patchSF[patchn])
                        count += 1
                    patchRF,countRF=dataRF(patchR,countR)
                    for patchn in range(countRF):
                        np.save(cropSubsetDir + '/' + subsetDirn + '/' + '1a' + '/' + str(count), patchRF[patchn])
                        count += 1
                    patchSR,countSR=dataSR(patchS, countS, newGeneratedCoor, voxelWidthZ, voxelWidthXY, newVolume)
                    for patchn in range(countSR):
                        np.save(cropSubsetDir + '/' + subsetDirn + '/' + '1a' + '/' + str(count), patchSR[patchn])
                        count += 1
                    patchSRF, countSRF = dataSRF(patchSR, countSR)
                    for patchn in range(countSRF):
                        np.save(cropSubsetDir + '/' + subsetDirn + '/' + '1a' + '/' + str(count), patchSRF[patchn])
                        count += 1


