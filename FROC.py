from sklearn import metrics
import matplotlib.pylab as plt
import numpy as np

testnum=[89,89, 89,89,89,89,89,89,88,88]


def FROC(labels,probs,index):
    # num of image
    totalNumberOfImages = testnum[index]
    numberOfDetectedLesions = sum(labels)
    totalNumberOfCandidates = len(probs)
    fpr, tpr, thresholds = metrics.roc_curve(labels, probs, pos_label=1)
    # FROC
    fps = fpr * (totalNumberOfCandidates - numberOfDetectedLesions) / totalNumberOfImages
    sens = tpr
    fps_itp = np.linspace(0.125, 8, num=10001)
    sens_itp = np.interp(fps_itp, fps, sens)
    frvvlu = 0
    nxth = 0.125
    fpps=[]
    for fp, ss in zip(fps_itp, sens_itp):
        if abs(fp - nxth) < 3e-4:
            print(ss)
            fpps.append(ss)
            frvvlu += ss
            nxth *= 2
        if abs(nxth - 16) < 1e-5: break
    print(frvvlu / 7, nxth)
    cpm=frvvlu / 7

    return fpps,cpm

if __name__ == '__main__':
    fppss = [0, 0, 0, 0, 0, 0, 0]
    cpms = 0
    for i in range(10):
        labels=np.load('data/label'+str(i)+'.npy')
        predict = np.load('data/pre' + str(i) + '.npy')
        fpps, cpm = FROC(labels, predict, i)
        cpms += cpm
        for k in range(7):
            fppss[k] += fpps[k]
    cpm_ave = cpms / 10
    fpps_ave = [i / 10 for i in fppss]
    print(fpps_ave, cpm_ave)


