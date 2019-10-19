import SimpleITK as sitk
import numpy as np
import csv
from scipy import ndimage


def load_itk_image(filename):
    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage)
    numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
    numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))

    return numpyImage, numpyOrigin, numpySpacing


def readCSV(filename):
    lines = []
    with open(filename, "r") as f:
        csvreader = csv.reader(f)
        for line in csvreader:
            lines.append(line)
    return lines


def worldToVoxelCoord(worldCoord, origin, spacing):
    strechedVoxelCoord = np.absolute(worldCoord - origin)
    voxelCoord = strechedVoxelCoord / spacing
    return voxelCoord


def dataNormalize(npzarray):
    maxHU = 400.  # maxHU = np.amax(npzarray)
    minHU = -1000.  # minHU = np.amin(npzarray)
    npzarray = (npzarray - minHU) / (maxHU - minHU)
    npzarray[npzarray > 1] = 1.
    npzarray[npzarray < 0] = 0.
    return npzarray


def dataFlip(arr):
    arrr = []
    arrr.append(np.flip(arr, 0))
    arrr.append(np.flip(arr, 1))
    arrr.append(np.flip(arr, 2))
    return arrr, 3


def dataRot(arr, newGeneratedCoor, voxelWidthZ, voxelWidthXY, newVolume):
    arrr = []
    count = 0
    arrr.append(np.rot90(arr, 2, (0, 1)))
    arrr.append(np.rot90(arr, 1, (2, 1)))
    arrr.append(np.rot90(arr, 2, (2, 1)))
    arrr.append(np.rot90(arr, 3, (2, 1)))
    arrr.append(np.rot90(arr, 2, (0, 2)))
    count += 5
    # 01
    patch = newVolume[
            int(newGeneratedCoor[0] - voxelWidthXY / 2):int(newGeneratedCoor[0] + voxelWidthXY / 2),
            int(newGeneratedCoor[1] - voxelWidthZ / 2):int(newGeneratedCoor[1] + voxelWidthZ / 2),
            int(newGeneratedCoor[2] - voxelWidthXY / 2):int(newGeneratedCoor[2] + voxelWidthXY / 2)]
    try:
        if np.shape(patch) != (voxelWidthZ, voxelWidthXY, voxelWidthXY):
            zoomFactor = [voxelWidthXY / float(np.shape(patch)[0]),
                          voxelWidthZ / float(np.shape(patch)[1]),
                          voxelWidthXY / float(np.shape(patch)[2])]
        patch = ndimage.zoom(patch, zoom=zoomFactor)
    except Exception:
        pass
    if np.shape(patch) == (voxelWidthZ, voxelWidthXY, voxelWidthXY):
        arrr.append(np.rot(patch, 1, (0, 1)))
        count += 1
        arrr.append(np.rot(patch, 3, (0, 1)))
        count += 1
    # 02
    patch = newVolume[
            int(newGeneratedCoor[0] - voxelWidthXY / 2):int(newGeneratedCoor[0] + voxelWidthXY / 2),
            int(newGeneratedCoor[1] - voxelWidthXY / 2):int(newGeneratedCoor[1] + voxelWidthXY / 2),
            int(newGeneratedCoor[2] - voxelWidthZ / 2):int(newGeneratedCoor[2] + voxelWidthZ / 2)]
    try:
        if np.shape(patch) != (voxelWidthZ, voxelWidthXY, voxelWidthXY):
            zoomFactor = [voxelWidthXY / float(np.shape(patch)[0]),
                          voxelWidthXY / float(np.shape(patch)[1]),
                          voxelWidthZ / float(np.shape(patch)[2])]
        patch = ndimage.zoom(patch, zoom=zoomFactor)
    except Exception:
        pass
    if np.shape(patch) == (voxelWidthZ, voxelWidthXY, voxelWidthXY):
        arrr.append(np.rot(patch, 1, (0, 2)))
        count += 1
        arrr.append(np.rot(patch, 3, (0, 2)))
        count += 1

    return arrr, count


def dataShift(newGeneratedCoor, voxelWidthZ, voxelWidthXY, newVolume):
    shift_range = [-2, 2]
    arrr = []
    countS = 0
    # 0
    for i in range(2):
        # 0
        patch = newVolume[
                int(newGeneratedCoor[0] - voxelWidthZ / 2 + shift_range[i]):int(
                    newGeneratedCoor[0] + voxelWidthZ / 2 + shift_range[i]),
                int(newGeneratedCoor[1] - voxelWidthXY / 2):int(newGeneratedCoor[1] + voxelWidthXY / 2),
                int(newGeneratedCoor[2] - voxelWidthXY / 2):int(newGeneratedCoor[2] + voxelWidthXY / 2)]
        try:
            if np.shape(patch) != (voxelWidthZ, voxelWidthXY, voxelWidthXY):
                zoomFactor = [voxelWidthZ / float(np.shape(patch)[0]),
                              voxelWidthXY / float(np.shape(patch)[1]),
                              voxelWidthXY / float(np.shape(patch)[2])]
            patch = ndimage.zoom(patch, zoom=zoomFactor)
        except Exception:
            pass
        if np.shape(patch) == (voxelWidthZ, voxelWidthXY, voxelWidthXY):
            arrr.append(patch)
            countS += 1

        # 1
        patch = newVolume[
                int(newGeneratedCoor[0] - voxelWidthZ / 2):int(newGeneratedCoor[0] + voxelWidthZ / 2),
                int(newGeneratedCoor[1] - voxelWidthXY / 2 + shift_range[i]):int(
                    newGeneratedCoor[1] + voxelWidthXY / 2 + shift_range[i]),
                int(newGeneratedCoor[2] - voxelWidthXY / 2):int(newGeneratedCoor[2] + voxelWidthXY / 2)]
        try:
            if np.shape(patch) != (voxelWidthZ, voxelWidthXY, voxelWidthXY):
                zoomFactor = [voxelWidthZ / float(np.shape(patch)[0]),
                              voxelWidthXY / float(np.shape(patch)[1]),
                              voxelWidthXY / float(np.shape(patch)[2])]
            patch = ndimage.zoom(patch, zoom=zoomFactor)
        except Exception:
            pass
        if np.shape(patch) == (voxelWidthZ, voxelWidthXY, voxelWidthXY):
            arrr.append(patch)
            countS += 1

        # 2
        patch = newVolume[
                int(newGeneratedCoor[0] - voxelWidthZ / 2):int(newGeneratedCoor[0] + voxelWidthZ / 2),
                int(newGeneratedCoor[1] - voxelWidthXY / 2):int(newGeneratedCoor[1] + voxelWidthXY / 2),
                int(newGeneratedCoor[2] - voxelWidthXY / 2 + shift_range[i]):int(
                    newGeneratedCoor[2] + voxelWidthXY / 2) + shift_range[i]]
        try:
            if np.shape(patch) != (voxelWidthZ, voxelWidthXY, voxelWidthXY):
                zoomFactor = [voxelWidthZ / float(np.shape(patch)[0]),
                              voxelWidthXY / float(np.shape(patch)[1]),
                              voxelWidthXY / float(np.shape(patch)[2])]
            patch = ndimage.zoom(patch, zoom=zoomFactor)
        except Exception:
            pass
        if np.shape(patch) == (voxelWidthZ, voxelWidthXY, voxelWidthXY):
            arrr.append(patch)
            countS += 1

    return arrr, countS


def dataSF(arrs, countS):
    arrr = []
    for countn in range(countS):
        arrr.append(np.flip(arrs[countn], 0))
        arrr.append(np.flip(arrs[countn], 1))
        arrr.append(np.flip(arrs[countn], 2))
    return arrr, 3 * countS

def dataRF(arrs, countR ):
    arrr = []
    for countn in range(countR):
        arrr.append(np.flip(arrs[countn], 0))
        arrr.append(np.flip(arrs[countn], 1))
        arrr.append(np.flip(arrs[countn], 2))
    return arrr, 3 * countR

def dataSR(arrs, countS, newGeneratedCoor, voxelWidthZ, voxelWidthXY, newVolume):
    arrr = []
    count = 0
    shift_range = [-2, 2]
    for countn in range(countS):
        arrr.append(np.rot90(arrs[countn], 2, (0, 1)))
        arrr.append(np.rot90(arrs[countn], 1, (2, 1)))
        arrr.append(np.rot90(arrs[countn], 2, (2, 1)))
        arrr.append(np.rot90(arrs[countn], 3, (2, 1)))
        arrr.append(np.rot90(arrs[countn], 2, (0, 2)))
        count += 5
    for i in range(2):
        # 010
        patch = newVolume[
                int(newGeneratedCoor[0] - voxelWidthXY / 2 + shift_range[i]):int(
                    newGeneratedCoor[0] + voxelWidthXY / 2 + shift_range[i]),
                int(newGeneratedCoor[1] - voxelWidthZ / 2):int(newGeneratedCoor[1] + voxelWidthZ / 2),
                int(newGeneratedCoor[2] - voxelWidthXY / 2):int(newGeneratedCoor[2] + voxelWidthXY / 2)]
        try:
            if np.shape(patch) != (voxelWidthZ, voxelWidthXY, voxelWidthXY):
                zoomFactor = [voxelWidthXY / float(np.shape(patch)[0]),
                              voxelWidthZ / float(np.shape(patch)[1]),
                              voxelWidthXY / float(np.shape(patch)[2])]
            patch = ndimage.zoom(patch, zoom=zoomFactor)
        except Exception:
            pass
        if np.shape(patch) == (voxelWidthZ, voxelWidthXY, voxelWidthXY):
            arrr.append(np.rot(patch, 1, (0, 1)))
            count += 1
            arrr.append(np.rot(patch, 3, (0, 1)))
            count += 1
        # 011
        patch = newVolume[
                int(newGeneratedCoor[0] - voxelWidthXY / 2):int(newGeneratedCoor[0] + voxelWidthXY / 2),
                int(newGeneratedCoor[1] - voxelWidthZ / 2 + shift_range[i]):int(
                    newGeneratedCoor[1] + voxelWidthZ / 2 + shift_range[i]),
                int(newGeneratedCoor[2] - voxelWidthXY / 2):int(newGeneratedCoor[2] + voxelWidthXY / 2)]
        try:
            if np.shape(patch) != (voxelWidthZ, voxelWidthXY, voxelWidthXY):
                zoomFactor = [voxelWidthXY / float(np.shape(patch)[0]),
                              voxelWidthZ / float(np.shape(patch)[1]),
                              voxelWidthXY / float(np.shape(patch)[2])]
            patch = ndimage.zoom(patch, zoom=zoomFactor)
        except Exception:
            pass
        if np.shape(patch) == (voxelWidthZ, voxelWidthXY, voxelWidthXY):
            arrr.append(np.rot(patch, 1, (0, 1)))
            count += 1
            arrr.append(np.rot(patch, 3, (0, 1)))
            count += 1
        # 012
        patch = newVolume[
                int(newGeneratedCoor[0] - voxelWidthXY / 2):int(newGeneratedCoor[0] + voxelWidthXY / 2),
                int(newGeneratedCoor[1] - voxelWidthZ / 2):int(newGeneratedCoor[1] + voxelWidthZ / 2 + shift_range[0]),
                int(newGeneratedCoor[2] - voxelWidthXY / 2 + shift_range[i]):int(
                    newGeneratedCoor[2] + voxelWidthXY / 2 + shift_range[i])]
        try:
            if np.shape(patch) != (voxelWidthZ, voxelWidthXY, voxelWidthXY):
                zoomFactor = [voxelWidthXY / float(np.shape(patch)[0]),
                              voxelWidthZ / float(np.shape(patch)[1]),
                              voxelWidthXY / float(np.shape(patch)[2])]
            patch = ndimage.zoom(patch, zoom=zoomFactor)
        except Exception:
            pass
        if np.shape(patch) == (voxelWidthZ, voxelWidthXY, voxelWidthXY):
            arrr.append(np.rot(patch, 1, (0, 1)))
            count += 1
            arrr.append(np.rot(patch, 3, (0, 1)))
            count += 1
        # 020
        patch = newVolume[
                int(newGeneratedCoor[0] - voxelWidthXY / 2 + shift_range[i]):int(
                    newGeneratedCoor[0] + voxelWidthXY / 2 + shift_range[i]),
                int(newGeneratedCoor[1] - voxelWidthXY / 2):int(newGeneratedCoor[1] + voxelWidthXY / 2),
                int(newGeneratedCoor[2] - voxelWidthZ / 2):int(newGeneratedCoor[2] + voxelWidthZ / 2)]
        try:
            if np.shape(patch) != (voxelWidthZ, voxelWidthXY, voxelWidthXY):
                zoomFactor = [voxelWidthXY / float(np.shape(patch)[0]),
                              voxelWidthXY / float(np.shape(patch)[1]),
                              voxelWidthZ / float(np.shape(patch)[2])]
            patch = ndimage.zoom(patch, zoom=zoomFactor)
        except Exception:
            pass
        if np.shape(patch) == (voxelWidthZ, voxelWidthXY, voxelWidthXY):
            arrr.append(np.rot(patch, 1, (0, 2)))
            count += 1
            arrr.append(np.rot(patch, 3, (0, 2)))
            count += 1
        # 021
        patch = newVolume[
                int(newGeneratedCoor[0] - voxelWidthXY / 2):int(newGeneratedCoor[0] + voxelWidthXY / 2),
                int(newGeneratedCoor[1] - voxelWidthXY / 2 + shift_range[i]):int(
                    newGeneratedCoor[1] + voxelWidthXY / 2 + shift_range[i]),
                int(newGeneratedCoor[2] - voxelWidthZ / 2):int(newGeneratedCoor[2] + voxelWidthZ / 2)]
        try:
            if np.shape(patch) != (voxelWidthZ, voxelWidthXY, voxelWidthXY):
                zoomFactor = [voxelWidthXY / float(np.shape(patch)[0]),
                              voxelWidthXY / float(np.shape(patch)[1]),
                              voxelWidthZ / float(np.shape(patch)[2])]
            patch = ndimage.zoom(patch, zoom=zoomFactor)
        except Exception:
            pass
        if np.shape(patch) == (voxelWidthZ, voxelWidthXY, voxelWidthXY):
            arrr.append(np.rot(patch, 1, (0, 2)))
            count += 1
            arrr.append(np.rot(patch, 3, (0, 2)))
            count += 1
        # 022
        patch = newVolume[
                int(newGeneratedCoor[0] - voxelWidthXY / 2):int(newGeneratedCoor[0] + voxelWidthXY / 2),
                int(newGeneratedCoor[1] - voxelWidthXY / 2):int(
                    newGeneratedCoor[1] + voxelWidthXY / 2),
                int(newGeneratedCoor[2] - voxelWidthZ / 2 + shift_range[i]):int(
                    newGeneratedCoor[2] + voxelWidthZ / 2 + shift_range[i])]
        try:
            if np.shape(patch) != (voxelWidthZ, voxelWidthXY, voxelWidthXY):
                zoomFactor = [voxelWidthXY / float(np.shape(patch)[0]),
                              voxelWidthXY / float(np.shape(patch)[1]),
                              voxelWidthZ / float(np.shape(patch)[2])]
            patch = ndimage.zoom(patch, zoom=zoomFactor)
        except Exception:
            pass
        if np.shape(patch) == (voxelWidthZ, voxelWidthXY, voxelWidthXY):
            arrr.append(np.rot(patch, 1, (0, 2)))
            count += 1
            arrr.append(np.rot(patch, 3, (0, 2)))
            count += 1

    return arrr, count

def dataSRF(arrs, countSR):
    arrr = []
    for countn in range(countSR):
        arrr.append(np.flip(arrs[countn], 0))
        arrr.append(np.flip(arrs[countn], 1))
        arrr.append(np.flip(arrs[countn], 2))
    return arrr, 3 * countSR