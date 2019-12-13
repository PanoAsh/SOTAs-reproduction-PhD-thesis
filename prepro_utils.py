import re, os, sys
import cv2
import matplotlib.pyplot as plt
import numpy as np

import settings


def file_rename(mode='left'):
    if mode == 'left':
        pathI = settings.L_PATH_RAW
        pathO = settings.L_PATH_TGT
    elif mode == 'right':
        pathI = settings.R_PATH_RAW
        pathO = settings.R_PATH_TGT
    else:
        print('No processing; Please check your input parameters.')

    filelist = os.listdir(pathI)
    filelist.sort(key=lambda x: x[:-4])

    count = 1
    for item in filelist:
        if item.endswith('.txt'):
            src = os.path.join(os.path.abspath(pathI), item)
            dst = os.path.join(os.path.abspath(pathO), format(str(count), '0>3s') + '.txt')
            try:
                os.rename(src, dst)
                print('converting %s to %s ...' % (src, dst))
            except:
                continue
        print(" {} images processed".format(count))
        count += 1

    print('Naming process done !')

def get_starts(fix_list):

    return np.where(fix_list[:, 0] == 0)[0]

def get_fixpos(fixationList, startPositions, scanpathIdx=0):
    if scanpathIdx >= startPositions.shape[0]-1:
        range_ = np.arange(startPositions[scanpathIdx], fixationList.shape[0])
    else:
        range_ = np.arange(startPositions[scanpathIdx], startPositions[scanpathIdx+1])

    print(range_, startPositions[scanpathIdx])

    return fixationList[range_, :].copy()


if __name__ == '__main__':
   print('waiting...')