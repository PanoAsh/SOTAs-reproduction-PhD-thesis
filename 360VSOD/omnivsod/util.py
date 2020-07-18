import os
import torch
from spherical_distortion.util import load_torch_img, torch2numpy
from spherical_distortion.functional import create_tangent_images, tangent_images_to_equirectangular
from torch.autograd import Variable
import torch.nn.functional as F
import cv2
import numpy as np


def listTrain():
    f1 = open(os.getcwd() + '/train_img.lst', 'w')
    f2 = open(os.getcwd() + '/train_msk.lst', 'w')
    f3 = open(os.getcwd() + '/test_img.lst', 'w')
    f4 = open(os.getcwd() + '/test_msk.lst', 'w')
    frm_count = 0
    for cls in os.listdir(os.getcwd() + '/data/train_img/'):
        for vid in os.listdir(os.path.join(os.getcwd() + '/data/train_img/', cls)):
            vidList = os.listdir(os.path.join(os.path.join(os.getcwd() + '/data/train_img/', cls), vid))
            vidList.sort(key=lambda x: x[:-4])
            for frm in vidList:
                lineImg = os.path.join(os.path.join(os.path.join(os.getcwd() + '/data/train_img/', cls), vid), frm) + \
                          '\n'
                lineMsk = os.path.join(os.path.join(os.path.join(os.getcwd() + '/data/train_msk/', cls), vid),
                                       'frame_' + frm[-10:]) + '\n'
                f1.write(lineImg)
                f2.write(lineMsk)
                frm_count += 1
    print(" {} frames wrote. ".format(frm_count))
    for cls in os.listdir(os.getcwd() + '/data/test_img/'):
        for vid in os.listdir(os.path.join(os.getcwd() + '/data/test_img/', cls)):
            vidList = os.listdir(os.path.join(os.path.join(os.getcwd() + '/data/test_img/', cls), vid))
            vidList.sort(key=lambda x: x[:-4])
            for frm in vidList:
                lineImg = os.path.join(os.path.join(os.path.join(os.getcwd() + '/data/test_img/', cls), vid), frm) + \
                          '\n'
                lineMsk = os.path.join(os.path.join(os.path.join(os.getcwd() + '/data/test_msk/', cls), vid),
                                       'frame_' + frm[-10:]) + '\n'
                f3.write(lineImg)
                f4.write(lineMsk)
                frm_count += 1
    print(" {} frames wrote. ".format(frm_count))
    f1.close()
    f2.close()
    f3.close()
    f4.close()

def ER2TI(ER, base_order, sample_order):
    #ER = ER.cuda() # not suggested for the dataload get_item process
    TIs = create_tangent_images(ER, base_order, sample_order)
    TIs = TIs.permute(1, 0, 2, 3)

    return TIs

def TI2ER(TIs, base_level, sample_level):
    ER = tangent_images_to_equirectangular(TIs, [int(2048 / 2 ** (10 - sample_level)),
                                                     int(4096 / 2 ** (10 - sample_level))],
                                           base_level, sample_level)

    return ER

def demo():
    img_pth = os.getcwd() + '/results_analysis/Img/'
    gt_pth = os.getcwd() + '/results_analysis/GT/'
    salEr_pth = os.getcwd() + '/results_analysis/Sal_ER/'
    salTi_pth = os.getcwd() + '/results_analysis/Sal_TI/'

    demo = cv2.VideoWriter(os.getcwd() + '/' + 'demo.avi', 0, 100, (1024, 512))
    img_list = os.listdir(img_pth)
    img_list.sort(key=lambda x: x[:-4])

    count = 1
    for item in img_list:
        img = cv2.imread(img_pth + item)
        gt = cv2.imread(gt_pth + item)
        salEr = cv2.imread(salEr_pth + item)
        salTi = cv2.imread(salTi_pth + item)

        frm = np.zeros((512, 1024, 3))
        frm[:256, :512, :] = img
        frm[:256, 512:, :] = gt
        frm[256:, :512, :] = salEr
        frm[256:, 512:, :] = salTi

        demo.write(np.uint8(frm))
        print("{} writen".format(count))
        count += 1

    demo.release()

def listTest():
    img_pth = os.getcwd() + '/results_analysis/Img/'
    img_list = os.listdir(img_pth)
    img_list.sort(key=lambda x: x[:-4])
    f = open(os.getcwd() + '/test.lst', 'w')

    for item in img_list:
        f.write(item + '\n')
    f.close()


if __name__ == '__main__':
    #listTrain()
    #ER2TI()
    #demo()
    #listTest()
    print()